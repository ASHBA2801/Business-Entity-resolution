"""Stage 2: LightGBM pair scorer that filters Stage-1 candidates down to a small,
hard top-K per Source-1 entity for the Stage-3 cross-encoder.

Two subcommands; every input and output is an explicit path (see README for the contract):

  train     Stage-1 detail for a training S1 file (train_split) and an evaluation S1 file
            (val) + source TSVs + ground truth -> model dir, filtered eval candidates,
            out-of-fold filtered training candidates (Stage-3 training input), reports
  predict   model dir + Stage-1 detail for any S1 file (e.g. test) -> filtered candidates

Data design
  * Training pairs = Stage-1 output for train_split S1 vs the train S2/S3 pool, i.e. the
    exact candidate generator that runs on test (no hand-made negatives).
  * Evaluation pairs = Stage-1 output for val S1 (disjoint S1 entities -> grouped split).
    Early stopping uses a further grouped holdout *inside* train_split, so the val
    threshold / filtering analysis is not also the early-stopping set.
  * Features are computed over train_split + val together, so cross-S1 competition
    features see every train S1, just as they see every test S1 at inference.

Model dir (written by train, read by predict):
  lgbm.txt, meta.json (features, filter settings), stage2_report.json,
  stage2_pr_curve.png, stage2_feature_importance.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import (DENSE_DERIVED, FEATURE_GROUPS, FEATURES,  # noqa: E402
                                 build_features, load_embeddings, load_records,
                                 load_vectorizers)
from utils import (default_n_jobs, ensure_parent, load_truth, read_tsv,  # noqa: E402
                   require_files, stage)

SOURCES = ("S2", "S3")


# =========================================================================== data
def build_universe(s1_sets: list[tuple], pool: dict, n_jobs: int):
    """Records (every S1 file + pool) and the concatenated Stage-1 pair detail.

    s1_sets: [(tag, s1_tsv, s1_emb_or_None, detail_parquet)]
    pool:    {"S2": (tsv, emb_or_None), "S3": (tsv, emb_or_None)}"""
    files = {f"S1:{tag}": path for tag, path, _, _ in s1_sets}
    files.update({s: pool[s][0] for s in SOURCES})
    rec = load_records(files, n_jobs)
    rows = {t: int((rec["emb_tag"] == t).sum()) for t in files}
    emb_paths = {f"S1:{tag}": emb for tag, _, emb, _ in s1_sets}
    emb_paths.update({s: pool[s][1] for s in SOURCES})
    embs = load_embeddings(emb_paths, rows)

    frames = []
    for tag, _, _, detail in s1_sets:
        d = pd.read_parquet(detail)
        d["split"] = tag
        frames.append(d)
    det = pd.concat(frames, ignore_index=True)
    det = det.drop(columns=[c for c in ("dense_score",) if c in det])
    for c in det.columns:
        if c.endswith("_rank"):
            det[c] = det[c].fillna(0)
    row = pd.Series(np.arange(len(rec)), index=rec.index)
    det["q"] = row.reindex(det["source1_entity_id"].to_numpy()).to_numpy()
    det["p"] = row.reindex(det["candidate_entity_id"].to_numpy()).to_numpy()
    if det[["q", "p"]].isna().any().any():
        raise ValueError("Stage-1 detail references IDs not in the given source files "
                         "(detail built from a different S1/S2/S3?)")
    det["q"] = det["q"].astype(np.int64)
    det["p"] = det["p"].astype(np.int64)
    det["cand_is_s3"] = det["candidate_entity_id"].str.startswith("S3-").astype(np.int8)
    det = det.sort_values(["q", "p"], kind="stable").reset_index(drop=True)
    return rec, det, embs


def s1_ids(path: str) -> list[str]:
    return read_tsv(path)["entity_id"].tolist()


def label_pairs(det: pd.DataFrame, truth: dict) -> np.ndarray:
    return np.fromiter((c in truth.get(s, ()) for s, c in
                        zip(det["source1_entity_id"], det["candidate_entity_id"])),
                       dtype=np.int8, count=len(det))


# =========================================================================== metrics
def f05(pred: set, true: set) -> float:
    """Per-entity F0.5 exactly as the challenge scores it (empty/empty = 1.0)."""
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: dict, truth: dict) -> float:
    return float(np.mean([f05(pred.get(s, set()), t) for s, t in truth.items()]))


def pair_threshold_table(scores, labels, n_true_total, thresholds) -> list[dict]:
    """Pair-level P / R / F0.5. Recall is end-to-end: denominator = ALL ground-truth
    pairs (including those Stage 1 never retrieved)."""
    out = []
    for t in thresholds:
        sel = scores >= t
        tp = int(labels[sel].sum())
        p = tp / max(int(sel.sum()), 1)
        r = tp / max(n_true_total, 1)
        f = 1.25 * p * r / (0.25 * p + r) if (p + r) else 0.0
        out.append({"threshold": t, "n_pred": int(sel.sum()), "precision": round(p, 4),
                    "recall": round(r, 4), "f05": round(f, 4)})
    return out


def entity_threshold_table(det, scores, truth, thresholds) -> list[dict]:
    """Macro F0.5 if 'score >= t' were used as the FINAL decision (singletons included)."""
    s1 = det["source1_entity_id"].to_numpy()
    cand = det["candidate_entity_id"].to_numpy()
    out = []
    for t in thresholds:
        sel = scores >= t
        pred = {}
        for a, b in zip(s1[sel], cand[sel]):
            pred.setdefault(a, set()).add(b)
        out.append({"threshold": t, "macro_f05": round(macro_f05(pred, truth), 4)})
    return out


# =========================================================================== filtering
def filter_candidates(det: pd.DataFrame, scores: np.ndarray, top_k: int, floor: float,
                      no_match: float) -> pd.DataFrame:
    """Per S1: if best score < no_match -> flag likely-no-match and pass nothing;
    otherwise pass up to top_k candidates with score >= floor (best first)."""
    d = pd.DataFrame({"s1": det["source1_entity_id"].to_numpy(),
                      "cand": det["candidate_entity_id"].to_numpy(), "score": scores})
    d = d.sort_values(["s1", "score"], ascending=[True, False], kind="stable")
    d["rk"] = d.groupby("s1", sort=False).cumcount() + 1
    best = d.groupby("s1", sort=False)["score"].transform("max")
    d["keep"] = (d["rk"] <= top_k) & (d["score"] >= floor) & (best >= no_match)
    return d


def filter_stats(fd: pd.DataFrame, truth: dict) -> dict:
    kept = fd[fd["keep"]]
    surv = kept.groupby("s1")["cand"].agg(set).to_dict()
    has_match = {s for s, t in truth.items() if t}
    singles = set(truth) - has_match
    n_true = sum(len(truth[s]) for s in has_match)
    retained = sum(len(surv.get(s, set()) & truth[s]) for s in has_match)
    lost_all = sum(1 for s in has_match if not (surv.get(s, set()) & truth[s]))
    passed_any = {s for s in truth if surv.get(s)}
    # oracle Stage 3 = perfect classifier over the surviving candidates
    oracle = macro_f05({s: surv.get(s, set()) & truth[s] for s in truth}, truth)
    return {
        "pairs_passed": int(len(kept)),
        "avg_pairs_per_s1": round(len(kept) / max(len(truth), 1), 3),
        "true_pairs_retained": round(retained / max(n_true, 1), 4),
        "entities_lost_all_true": int(lost_all),
        "entities_lost_all_true_pct": round(lost_all / max(len(has_match), 1), 4),
        "singletons_flagged_empty": round(len(singles - passed_any) / max(len(singles), 1), 4),
        "matched_flagged_empty": int(len(has_match - passed_any)),
        "oracle_stage3_macro_f05": round(oracle, 4),
    }


def write_filtered(path: str, fd: pd.DataFrame, all_s1: list[str]):
    kept = fd[fd["keep"]]
    by = {s: g for s, g in kept.groupby("s1", sort=False)}
    best = fd.groupby("s1")["score"].max().to_dict()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\tcandidate_scores\t"
                "best_score\tlikely_no_match\n")
        for s in all_s1:
            g = by.get(s)
            ids = ",".join(g["cand"]) if g is not None else ""
            sc = ",".join(f"{x:.4f}" for x in g["score"]) if g is not None else ""
            b = best.get(s, 0.0)
            f.write(f"{s}\t{ids}\t{sc}\t{b:.4f}\t{int(g is None)}\n")
    os.replace(tmp, path)


# =========================================================================== model
def _lgbm_params(cfg) -> dict:
    params = dict(objective="binary", learning_rate=cfg.lr, num_leaves=cfg.num_leaves,
                  min_child_samples=cfg.min_child_samples, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  metric=["average_precision", "binary_logloss"], verbose=-1,
                  num_threads=cfg.n_jobs, seed=cfg.seed)
    if cfg.scale_pos_weight != 1.0:
        params["scale_pos_weight"] = cfg.scale_pos_weight
    return params


def oof_scores(X, y, s1col, is_tr, m_fit, m_es, model, cfg) -> np.ndarray:
    """Stage-2 scores for every train_split pair from a model that never trained on that
    pair's S1, so Stage 3 trains on the same kind of top-K it sees at test time.
    S1s used for fitting/early stopping are cross-fitted in grouped folds (fixed rounds =
    the main model's best iteration); unsampled S1s (--max-train-s1) were never seen and
    are scored by the main model."""
    import lightgbm as lgb
    scores = model.predict(X[is_tr], num_iteration=model.best_iteration)
    tr_rows = np.flatnonzero(is_tr)
    used = m_fit | m_es
    groups = s1col[used].unique()
    rng = np.random.default_rng(cfg.seed + 1)
    fold_of = pd.Series(rng.permutation(len(groups)) % cfg.oof_folds, index=groups)
    fold = np.full(len(s1col), -1)
    fold[used] = fold_of.reindex(s1col[used].to_numpy()).to_numpy()
    pos_in_tr = np.full(len(s1col), -1)
    pos_in_tr[tr_rows] = np.arange(len(tr_rows))
    for k in range(cfg.oof_folds):
        m_train, m_pred = used & (fold != k), fold == k
        dtr = lgb.Dataset(X[m_train], y[m_train], feature_name=list(X.columns))
        mk = lgb.train(_lgbm_params(cfg), dtr, num_boost_round=model.best_iteration)
        scores[pos_in_tr[m_pred]] = mk.predict(X[m_pred])
        print(f"    fold {k + 1}/{cfg.oof_folds}: trained on {m_train.sum():,} pairs, "
              f"scored {m_pred.sum():,}", flush=True)
    return scores


def train_lgbm(X_tr, y_tr, X_es, y_es, cfg, features):
    import lightgbm as lgb
    params = _lgbm_params(cfg)
    dtr = lgb.Dataset(X_tr, y_tr, feature_name=features, free_raw_data=False)
    des = lgb.Dataset(X_es, y_es, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=cfg.num_rounds, valid_sets=[des],
                      valid_names=["es"],
                      callbacks=[lgb.early_stopping(cfg.early_stopping, first_metric_only=True,
                                                    verbose=False),
                                 lgb.log_evaluation(0)])
    return model


def average_precision(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if y.sum() else float("nan")


def plot_pr(path, val_tab, val_scores, val_labels, n_true_total, chosen_t):
    """Two panels (never a dual axis): pair-level PR curve, and P/R/F0.5 vs threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    p, r, _ = precision_recall_curve(val_labels, val_scores)
    r = r * val_labels.sum() / max(n_true_total, 1)  # end-to-end recall
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.grid(color=grid, linewidth=0.8)
        a.tick_params(colors=muted)
        for sp in a.spines.values():
            sp.set_visible(False)
    ax[0].plot(r, p, color=blue, linewidth=2)
    best = max(val_tab, key=lambda x: x["f05"])
    ax[0].plot(best["recall"], best["precision"], "o", color=blue, markersize=8,
               markeredgecolor="#fcfcfb", markeredgewidth=2)
    ax[0].annotate(f"max F0.5 {best['f05']:.3f} @ t={best['threshold']}",
                   (best["recall"], best["precision"]), xytext=(-150, -30),
                   textcoords="offset points", color=ink, fontsize=9)
    ax[0].set_xlabel("Recall (end-to-end, incl. Stage-1 misses)", color=muted)
    ax[0].set_ylabel("Precision", color=muted)
    ax[0].set_title("Pair-level precision-recall (val)", color=ink, loc="left")
    t = [x["threshold"] for x in val_tab]
    for key, col, lab in (("precision", blue, "Precision"), ("recall", orange, "Recall"),
                          ("f05", aqua, "F0.5")):
        y = [x[key] for x in val_tab]
        ax[1].plot(t, y, color=col, linewidth=2, marker="o", markersize=4, label=lab)
    ax[1].axvline(chosen_t, color=muted, linewidth=1, linestyle="--")
    ax[1].annotate("no-match threshold", (chosen_t, 0), xytext=(4, 4),
                   textcoords="offset points", color=muted, fontsize=8,
                   xycoords=("data", "axes fraction"))
    ax[1].set_xlabel("LightGBM score threshold", color=muted)
    ax[1].set_title("Pair-level metrics vs threshold (val)", color=ink, loc="left")
    ax[1].legend(frameon=False, loc="best", labelcolor=ink)
    lo = min(min(x[k] for x in val_tab) for k in ("precision", "recall", "f05"))
    ax[1].set_ylim(max(0.0, lo - 0.05 * (1.0 - lo) - 0.01), 1.0 + 0.02 * (1.0 - lo) + 0.002)
    lo0 = max(0.0, min(0.9, best["precision"] - 0.25))
    ax[0].set_ylim(lo0, 1.005)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =========================================================================== main
def pool_arg(cfg) -> dict:
    return {s: (getattr(cfg, s.lower()), getattr(cfg, f"{s.lower()}_emb")) for s in SOURCES}


def run_train(cfg):
    t0 = time.perf_counter()
    rep = {"config": vars(cfg).copy()}
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    require_files(cfg.train_s1, cfg.train_detail, cfg.train_truth, cfg.eval_s1,
                  cfg.eval_detail, cfg.eval_truth, cfg.s2, cfg.s3, cfg.vectorizers)
    if cfg.oof_output and cfg.oof_folds < 2:
        raise SystemExit("--oof-output needs --oof-folds >= 2")
    os.makedirs(cfg.model_dir, exist_ok=True)

    with stage("load records + Stage-1 pairs (train + eval S1)"):
        rec, det, embs = build_universe(
            [("train", cfg.train_s1, cfg.train_s1_emb, cfg.train_detail),
             ("eval", cfg.eval_s1, cfg.eval_s1_emb, cfg.eval_detail)], pool_arg(cfg), cfg.n_jobs)
        truth = {"train": load_truth(cfg.train_truth), "eval": load_truth(cfg.eval_truth)}
        det["label"] = label_pairs(det, {**truth["train"], **truth["eval"]})
        print(f"    records {len(rec):,} | pairs {len(det):,} | embeddings {sorted(embs)}")

    with stage("features"):
        X = build_features(det, rec, load_vectorizers(cfg.vectorizers), embs, cfg.n_jobs)
        print(f"    {X.shape[1]} features | NaN share bge_m3_cos "
              f"{X['bge_m3_cos'].isna().mean():.3f}")

    is_tr = (det["split"] == "train").to_numpy()
    is_ev = ~is_tr
    y = det["label"].to_numpy()

    # ---- class balance -------------------------------------------------------------
    bal = {}
    for name, m in (("train", is_tr), ("val", is_ev)):
        pos, n = int(y[m].sum()), int(m.sum())
        bal[name] = {"pairs": n, "pos": pos, "neg": n - pos,
                     "pos_rate": round(pos / max(n, 1), 4),
                     "neg_per_pos": round((n - pos) / max(pos, 1), 2)}
    rep["class_balance"] = bal
    print("\nCLASS BALANCE")
    for k, v in bal.items():
        print(f"  {k:<5} pairs {v['pairs']:>9,} | pos {v['pos']:>8,} | neg {v['neg']:>9,} | "
              f"pos rate {v['pos_rate']:.4f} | neg:pos {v['neg_per_pos']}:1")

    # ---- grouped early-stopping holdout inside train_split ------------------------
    rng = np.random.default_rng(cfg.seed)
    tr_s1 = det.loc[is_tr, "source1_entity_id"].unique()
    if cfg.max_train_s1 and len(tr_s1) > cfg.max_train_s1:
        tr_s1 = rng.choice(tr_s1, cfg.max_train_s1, replace=False)
    es_s1 = set(rng.choice(tr_s1, int(len(tr_s1) * cfg.es_frac), replace=False))
    s1col = det["source1_entity_id"]
    m_es = is_tr & s1col.isin(es_s1).to_numpy()
    m_fit = is_tr & s1col.isin(set(tr_s1)).to_numpy() & ~m_es
    assert not (set(s1col[m_fit]) & set(s1col[m_es])) and not (set(s1col[m_fit]) & set(s1col[is_ev]))
    print(f"\n  grouped split: fit {m_fit.sum():,} pairs / {s1col[m_fit].nunique():,} S1 | "
          f"early-stop {m_es.sum():,} / {s1col[m_es].nunique():,} S1 | "
          f"val {is_ev.sum():,} / {s1col[is_ev].nunique():,} S1")

    with stage("train LightGBM"):
        model = train_lgbm(X[m_fit], y[m_fit], X[m_es], y[m_es], cfg, FEATURES)
        print(f"    best iteration {model.best_iteration}")
        s_val = model.predict(X[is_ev], num_iteration=model.best_iteration)
        rep["val_average_precision"] = round(average_precision(y[is_ev], s_val), 4)
        rep["best_iteration"] = model.best_iteration

    # ---- ablation: is the dense signal (Feature 58) adding anything? ----------------
    if cfg.ablation:
        with stage("ablation: no dense-derived features"):
            nod = [f for f in FEATURES if f not in DENSE_DERIVED]
            m2 = train_lgbm(X.loc[m_fit, nod], y[m_fit], X.loc[m_es, nod], y[m_es], cfg, nod)
            s2 = m2.predict(X.loc[is_ev, nod], num_iteration=m2.best_iteration)
            rep["ablation_no_dense"] = {"val_average_precision": round(average_precision(y[is_ev], s2), 4)}

    # ---- feature importance --------------------------------------------------------
    gain = model.feature_importance("gain")
    split_cnt = model.feature_importance("split")
    imp = pd.DataFrame({"feature": FEATURES, "gain": gain, "split": split_cnt})
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
    grp_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    imp["group"] = imp["feature"].map(grp_of)
    imp["dense_derived"] = imp["feature"].isin(DENSE_DERIVED)
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp["gain_rank"] = np.arange(1, len(imp) + 1)
    imp.to_csv(os.path.join(cfg.model_dir, "stage2_feature_importance.tsv"), sep="\t", index=False)
    rep["importance_by_group_pct"] = imp.groupby("group")["gain_pct"].sum().round(2).to_dict()
    rep["dense_derived_gain_pct"] = round(float(imp.loc[imp["dense_derived"], "gain_pct"].sum()), 2)
    b = imp[imp["feature"] == "bge_m3_cos"].iloc[0]
    rep["bge_m3_cos"] = {"gain_rank": int(b["gain_rank"]), "gain_pct": round(float(b["gain_pct"]), 2)}

    # ---- thresholds on val ---------------------------------------------------------
    val_det = det[is_ev].reset_index(drop=True)
    vtruth = truth["eval"]
    n_true_total = sum(len(t) for t in vtruth.values())
    ptab = pair_threshold_table(s_val, y[is_ev], n_true_total, thresholds)
    etab = entity_threshold_table(val_det, s_val, vtruth, thresholds)
    for a, b_ in zip(ptab, etab):
        a["macro_f05_if_final"] = b_["macro_f05"]
    rep["val_thresholds"] = ptab
    best_pair = max(ptab, key=lambda x: x["f05"])
    rep["val_max_pair_f05"] = best_pair
    rep["stage1_val_recall_ceiling"] = round(int(y[is_ev].sum()) / max(n_true_total, 1), 4)

    # ---- filtering sweep -------------------------------------------------------------
    sweep = []
    for k in cfg.sweep_k:
        for fl in cfg.sweep_floor:
            st = filter_stats(filter_candidates(val_det, s_val, k, fl, fl), vtruth)
            sweep.append({"top_k": k, "floor=no_match": fl, **st})
    rep["filter_sweep"] = sweep
    fd = filter_candidates(val_det, s_val, cfg.top_k, cfg.score_floor, cfg.no_match_threshold)
    rep["filter_chosen"] = {"top_k": cfg.top_k, "score_floor": cfg.score_floor,
                            "no_match_threshold": cfg.no_match_threshold,
                            **filter_stats(fd, vtruth)}
    # where were true matches lost?  stage 1 (never a candidate) vs stage 2 (filtered)
    cand_sets = val_det.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()
    surv = fd[fd["keep"]].groupby("s1")["cand"].agg(set).to_dict()
    lost1 = lost2 = 0
    for s, t in vtruth.items():
        if not t:
            continue
        if not (cand_sets.get(s, set()) & t):
            lost1 += 1
        elif not (surv.get(s, set()) & t):
            lost2 += 1
    ntruemax = sum(1 for t in vtruth.values() if t)
    rep["lost_true_match_entities"] = {"matched_entities": ntruemax,
                                       "lost_at_stage1": lost1, "lost_at_stage2_filter": lost2}
    mcount = pd.Series([len(t) for t in vtruth.values()])
    rep["val_matches_per_s1"] = {str(k): int(v) for k, v in
                                 mcount.clip(upper=8).value_counts().sort_index().items()}

    with stage("write outputs"):
        model.save_model(os.path.join(cfg.model_dir, "lgbm.txt"),
                         num_iteration=model.best_iteration)
        with open(os.path.join(cfg.model_dir, "meta.json"), "w") as f:
            json.dump({"features": FEATURES, "best_iteration": model.best_iteration,
                       "top_k": cfg.top_k, "score_floor": cfg.score_floor,
                       "no_match_threshold": cfg.no_match_threshold,
                       "trained_with_embeddings": sorted(embs)}, f, indent=2)
        write_filtered(ensure_parent(cfg.eval_output), fd, s1_ids(cfg.eval_s1))
        plot_pr(os.path.join(cfg.model_dir, "stage2_pr_curve.png"), ptab, s_val, y[is_ev],
                n_true_total, cfg.no_match_threshold)
        print(f"    wrote {cfg.model_dir}/lgbm.txt, {cfg.eval_output}")

    # ---- out-of-fold training candidates (Stage-3 training pairs) ----------------------
    if cfg.oof_output:
        with stage(f"out-of-fold filtered candidates ({cfg.oof_folds} folds)"):
            s_oof = oof_scores(X, y, s1col, is_tr, m_fit, m_es, model, cfg)
            tr_det = det[is_tr].reset_index(drop=True)
            ofd = filter_candidates(tr_det, s_oof, cfg.top_k, cfg.score_floor, cfg.no_match_threshold)
            write_filtered(ensure_parent(cfg.oof_output), ofd, s1_ids(cfg.train_s1))
            # should match the val filter stats if OOF scores mimic the deployed model
            rep["oof_train"] = {"folds": cfg.oof_folds,
                                "average_precision": round(average_precision(y[is_tr], s_oof), 4),
                                **filter_stats(ofd, truth["train"])}
            print(f"    OOF AP {rep['oof_train']['average_precision']} "
                  f"(val AP {rep['val_average_precision']}) | "
                  f"{rep['oof_train']['avg_pairs_per_s1']} pairs/S1 "
                  f"(val {rep['filter_chosen']['avg_pairs_per_s1']}) -> {cfg.oof_output}")

    rep["runtime_s"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(cfg.model_dir, "stage2_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print_report(rep, imp)
    return rep


def run_predict(cfg):
    """Score + filter the Stage-1 candidates of one S1 file with a trained model dir."""
    import lightgbm as lgb
    t0 = time.perf_counter()
    mpath, meta_path = (os.path.join(cfg.model_dir, f) for f in ("lgbm.txt", "meta.json"))
    require_files(mpath, meta_path, cfg.s1, cfg.detail, cfg.s2, cfg.s3, cfg.vectorizers, cfg.truth)
    meta = json.load(open(meta_path))
    if meta["features"] != FEATURES:
        raise ValueError(f"{meta_path}: model features differ from this code's FEATURES")
    top_k = cfg.top_k if cfg.top_k is not None else meta["top_k"]
    floor = cfg.score_floor if cfg.score_floor is not None else meta["score_floor"]
    no_match = cfg.no_match_threshold if cfg.no_match_threshold is not None else meta["no_match_threshold"]

    with stage("load records + Stage-1 pairs"):
        rec, det, embs = build_universe([("q", cfg.s1, cfg.s1_emb, cfg.detail)], pool_arg(cfg),
                                        cfg.n_jobs)
        if meta.get("trained_with_embeddings") and len(embs) < 3:
            print("    WARNING: model was trained with BGE-M3 embeddings but not all of "
                  "--s1-emb/--s2-emb/--s3-emb are usable here; bge_m3_cos will be NaN", flush=True)
    with stage("features + score"):
        X = build_features(det, rec, load_vectorizers(cfg.vectorizers), embs, cfg.n_jobs)
        scores = lgb.Booster(model_file=mpath).predict(X)
    with stage("filter + write"):
        fd = filter_candidates(det, scores, top_k, floor, no_match)
        ids = s1_ids(cfg.s1)
        write_filtered(ensure_parent(cfg.output), fd, ids)
    kept = fd[fd["keep"]]
    ck = rec["ckey"].to_numpy()[det["q"].to_numpy()]
    rep = {"s1": cfg.s1, "n_s1": len(ids), "pairs_in": int(len(det)),
           "pairs_passed": int(len(kept)),
           "likely_no_match_s1": int(len(ids) - kept["s1"].nunique()),
           "filter": {"top_k": top_k, "score_floor": floor, "no_match_threshold": no_match},
           "mean_score_by_country": pd.Series(scores).groupby(ck).mean().round(4).to_dict(),
           "output": cfg.output}
    if cfg.truth:
        truth = load_truth(cfg.truth)
        rep["average_precision"] = round(average_precision(label_pairs(det, truth), scores), 4)
        rep["filter_stats"] = filter_stats(fd, truth)
    rep["runtime_s"] = round(time.perf_counter() - t0, 1)
    if cfg.report:
        with open(ensure_parent(cfg.report), "w") as f:
            json.dump(rep, f, indent=2, default=float)
    print(json.dumps(rep, indent=2, default=float), flush=True)
    return rep


def print_report(rep, imp):
    print("\n" + "=" * 78 + "\nSTAGE 2 (LightGBM filter) SUMMARY\n" + "=" * 78)
    print(f"best iteration {rep['best_iteration']} | val average precision "
          f"{rep['val_average_precision']}"
          + (f" | without dense-derived features {rep['ablation_no_dense']['val_average_precision']}"
             if "ablation_no_dense" in rep else ""))
    print("\nFEATURE IMPORTANCE (gain %, top 20)")
    for _, r in imp.head(20).iterrows():
        tag = "  <- Feature 58 (BGE-M3)" if r["feature"] == "bge_m3_cos" else (
            "  (dense-derived)" if r["dense_derived"] else "")
        print(f"  {r['gain_rank']:>2}. {r['feature']:<20} {r['gain_pct']:6.2f}%  [{r['group']}]{tag}")
    print(f"  bge_m3_cos: rank {rep['bge_m3_cos']['gain_rank']}/{len(imp)}, "
          f"{rep['bge_m3_cos']['gain_pct']}% | all dense-derived {rep['dense_derived_gain_pct']}%")
    print("  by group: " + " | ".join(f"{k} {v}%" for k, v in
                                      sorted(rep["importance_by_group_pct"].items(), key=lambda x: -x[1])))
    print(f"\nVAL THRESHOLDS (pair-level; recall end-to-end, Stage-1 ceiling "
          f"{rep['stage1_val_recall_ceiling']})")
    print(f"  {'thr':>5} {'n_pred':>8} {'prec':>7} {'recall':>7} {'F0.5':>7} {'macroF0.5*':>11}")
    for r in rep["val_thresholds"]:
        print(f"  {r['threshold']:>5} {r['n_pred']:>8,} {r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['f05']:>7.4f} {r['macro_f05_if_final']:>11.4f}")
    print("  * entity macro F0.5 if 'score >= thr' were the final decision (singletons incl.)")
    print("\nFILTER SWEEP (val; floor used as both per-candidate floor and no-match threshold)")
    print(f"  {'K':>3} {'floor':>6} {'pairs/S1':>8} {'truePairsKept':>13} {'lostAll':>8} "
          f"{'singleEmpty':>11} {'oracleF0.5':>10}")
    for r in rep["filter_sweep"]:
        print(f"  {r['top_k']:>3} {r['floor=no_match']:>6} {r['avg_pairs_per_s1']:>8} "
              f"{r['true_pairs_retained']:>13} {r['entities_lost_all_true']:>8} "
              f"{r['singletons_flagged_empty']:>11} {r['oracle_stage3_macro_f05']:>10}")
    c = rep["filter_chosen"]
    lt = rep["lost_true_match_entities"]
    print(f"\nCHOSEN FILTER: top_k={c['top_k']} floor={c['score_floor']} "
          f"no_match={c['no_match_threshold']} -> {c['avg_pairs_per_s1']} pairs/S1, "
          f"{c['true_pairs_retained']} of true pairs kept, oracle macro F0.5 {c['oracle_stage3_macro_f05']}")
    print(f"TRUE MATCH LOST ENTIRELY: {lt['lost_at_stage2_filter']} S1 entities lost at the "
          f"Stage-2 filter (+{lt['lost_at_stage1']} already lost at Stage 1) "
          f"out of {lt['matched_entities']} S1 with >=1 true match")
    print(f"val true matches per S1 (8 = 8+): {rep['val_matches_per_s1']}")
    print(f"runtime {rep['runtime_s']}s\n" + "=" * 78, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-jobs", type=int, default=default_n_jobs())
    pool = common.add_argument_group("pool inputs (the S2/S3 files Stage 1 matched against)")
    pool.add_argument("--s2", required=True, help="Source-2 TSV")
    pool.add_argument("--s3", required=True, help="Source-3 TSV")
    pool.add_argument("--s2-emb", help="Stage-1 BGE-M3 embeddings of --s2 (.npy; optional)")
    pool.add_argument("--s3-emb", help="Stage-1 BGE-M3 embeddings of --s3 (.npy; optional)")
    pool.add_argument("--vectorizers", required=True, help="Stage-1 TF-IDF vectorizers pickle")

    t = sub.add_parser("train", parents=[common], help="train the LightGBM filter")
    i = t.add_argument_group("training / evaluation inputs")
    i.add_argument("--train-s1", required=True, help="training Source-1 TSV (train_split)")
    i.add_argument("--train-truth", required=True, help="ground truth for --train-s1")
    i.add_argument("--train-detail", required=True, help="Stage-1 detail parquet for --train-s1")
    i.add_argument("--train-s1-emb", help="Stage-1 embeddings of --train-s1 (optional)")
    i.add_argument("--eval-s1", required=True, help="evaluation Source-1 TSV (val)")
    i.add_argument("--eval-truth", required=True, help="ground truth for --eval-s1")
    i.add_argument("--eval-detail", required=True, help="Stage-1 detail parquet for --eval-s1")
    i.add_argument("--eval-s1-emb", help="Stage-1 embeddings of --eval-s1 (optional)")
    o = t.add_argument_group("outputs")
    o.add_argument("--model-dir", required=True, help="model + meta + reports directory")
    o.add_argument("--eval-output", required=True,
                   help="filtered candidates TSV for --eval-s1 (Stage-3 validation input)")
    o.add_argument("--oof-output",
                   help="out-of-fold filtered candidates TSV for --train-s1 (Stage-3 "
                        "training input)")
    t.add_argument("--oof-folds", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-train-s1", type=int, default=0, help="subsample training S1 (0 = all)")
    t.add_argument("--es-frac", type=float, default=0.15, help="grouped early-stopping holdout")
    # model
    t.add_argument("--lr", type=float, default=0.05)
    t.add_argument("--num-leaves", type=int, default=63)
    t.add_argument("--min-child-samples", type=int, default=50)
    t.add_argument("--num-rounds", type=int, default=3000)
    t.add_argument("--early-stopping", type=int, default=100)
    t.add_argument("--scale-pos-weight", type=float, default=1.0)
    t.add_argument("--ablation", action=argparse.BooleanOptionalAction, default=True)
    # filtering (stored in meta.json and reused by predict)
    t.add_argument("--top-k", type=int, default=3)
    t.add_argument("--score-floor", type=float, default=0.02)
    t.add_argument("--no-match-threshold", type=float, default=0.05)
    t.add_argument("--sweep-k", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10])
    t.add_argument("--sweep-floor", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])

    r = sub.add_parser("predict", parents=[common], help="filter one S1 file's candidates")
    i = r.add_argument_group("inputs")
    i.add_argument("--model-dir", required=True, help="directory written by train")
    i.add_argument("--s1", required=True, help="Source-1 TSV")
    i.add_argument("--detail", required=True, help="Stage-1 detail parquet for --s1")
    i.add_argument("--s1-emb", help="Stage-1 embeddings of --s1 (optional)")
    i.add_argument("--truth", help="ground truth for --s1 (optional: adds filter stats)")
    o = r.add_argument_group("outputs")
    o.add_argument("--output", required=True, help="filtered candidates TSV")
    o.add_argument("--report", help="report JSON")
    r.add_argument("--top-k", type=int, default=None, help="default: value in meta.json")
    r.add_argument("--score-floor", type=float, default=None, help="default: value in meta.json")
    r.add_argument("--no-match-threshold", type=float, default=None,
                   help="default: value in meta.json")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    (run_train if args.cmd == "train" else run_predict)(args)
