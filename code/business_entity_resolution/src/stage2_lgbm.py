"""Stage 2: LightGBM pair scorer that filters Stage-1 candidates down to a small,
hard top-K per Source-1 entity for the Stage-3 cross-encoder.

Usage (from code/business_entity_resolution/):
    # prerequisites: Stage 1 run with the same --cache-dir for every split used
    python src/blocking_pipeline.py --queries train_split   # training pairs
    python src/blocking_pipeline.py --queries val           # held-out evaluation pairs
    python src/blocking_pipeline.py --queries test          # (optional) pairs to filter
    python src/stage2_lgbm.py

Data design
  * Training pairs = Stage-1 output for train_split S1 vs the train S2/S3 pool, i.e. the
    exact candidate generator that runs on test (no hand-made negatives).
  * Evaluation pairs = Stage-1 output for val S1 (disjoint S1 entities -> grouped split).
    Early stopping uses a further grouped holdout *inside* train_split, so the val
    threshold / filtering analysis is not also the early-stopping set.
  * Features are computed over train_split + val together, so cross-S1 competition
    features see every train S1, just as they see every test S1 at inference.

Outputs (--output-dir):
  stage2_lgbm.txt                  LightGBM model   (+ stage2_meta.json: features, config)
  val_filtered_candidates.tsv      filtered val candidates (for Stage-3 development)
  filtered_candidates.tsv          filtered test candidates (if test Stage-1 detail exists)
  stage2_report.json, stage2_pr_curve.png, stage2_feature_importance.tsv
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
from utils import default_n_jobs, read_tsv, stage  # noqa: E402


# =========================================================================== data
def s1_path(data_dir, split):
    return os.path.join(data_dir, split, f"{split}_source1.tsv")


def pool_paths(data_dir, pool_split):
    return {s: os.path.join(data_dir, pool_split, f"{pool_split}_source{s[1]}.tsv")
            for s in ("S2", "S3")}


def build_universe(cfg, s1_splits: list[str], pool_split: str):
    """Records (S1 of every split + pool) and the concatenated Stage-1 pair detail."""
    files = {f"S1:{sp}": s1_path(cfg.data_dir, sp) for sp in s1_splits}
    files.update(pool_paths(cfg.data_dir, pool_split))
    rec = load_records(files, cfg.n_jobs)
    rows = {t: int((rec["emb_tag"] == t).sum()) for t in files}
    emb_map = {f"S1:{sp}": (sp, "S1") for sp in s1_splits}
    emb_map.update({s: (pool_split, s) for s in ("S2", "S3")})
    embs = load_embeddings(cfg.cache_dir, emb_map, rows)

    frames = []
    for sp in s1_splits:
        p = os.path.join(cfg.cache_dir, f"{sp}_candidates_detail.parquet")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing: run blocking_pipeline.py --queries {sp} "
                                    f"--cache-dir {cfg.cache_dir}")
        d = pd.read_parquet(p)
        d["split"] = sp
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
        raise ValueError("Stage-1 detail references IDs not in the source files")
    det["q"] = det["q"].astype(np.int64)
    det["p"] = det["p"].astype(np.int64)
    det["cand_is_s3"] = det["candidate_entity_id"].str.startswith("S3-").astype(np.int8)
    det = det.sort_values(["q", "p"], kind="stable").reset_index(drop=True)
    return rec, det, embs


def load_truth(data_dir, split) -> dict[str, set]:
    gt = read_tsv(os.path.join(data_dir, split, f"{split}_ground_truth.tsv"))
    return {s: set(m.split(",")) - {""} for s, m in
            zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


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
def run(cfg):
    t0 = time.perf_counter()
    rep = {"config": vars(cfg).copy()}
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    with stage("load records + Stage-1 pairs (train_split + val)"):
        splits = [cfg.train_split, cfg.eval_split]
        rec, det, embs = build_universe(cfg, splits, "train")
        truth = {sp: load_truth(cfg.data_dir, sp) for sp in splits}
        all_truth = {**truth[cfg.train_split], **truth[cfg.eval_split]}
        det["label"] = label_pairs(det, all_truth)
        print(f"    records {len(rec):,} | pairs {len(det):,} | dense caches {sorted(embs)}")

    with stage("features"):
        X = build_features(det, rec, load_vectorizers(cfg.cache_dir), embs, cfg.n_jobs)
        print(f"    {X.shape[1]} features | NaN share bge_m3_cos "
              f"{X['bge_m3_cos'].isna().mean():.3f}")

    is_tr = (det["split"] == cfg.train_split).to_numpy()
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
    os.makedirs(cfg.output_dir, exist_ok=True)
    imp.to_csv(os.path.join(cfg.output_dir, "stage2_feature_importance.tsv"), sep="\t", index=False)
    rep["importance_by_group_pct"] = imp.groupby("group")["gain_pct"].sum().round(2).to_dict()
    rep["dense_derived_gain_pct"] = round(float(imp.loc[imp["dense_derived"], "gain_pct"].sum()), 2)
    b = imp[imp["feature"] == "bge_m3_cos"].iloc[0]
    rep["bge_m3_cos"] = {"gain_rank": int(b["gain_rank"]), "gain_pct": round(float(b["gain_pct"]), 2)}

    # ---- thresholds on val ---------------------------------------------------------
    val_det = det[is_ev].reset_index(drop=True)
    vtruth = truth[cfg.eval_split]
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
        model.save_model(os.path.join(cfg.output_dir, "stage2_lgbm.txt"),
                         num_iteration=model.best_iteration)
        with open(os.path.join(cfg.output_dir, "stage2_meta.json"), "w") as f:
            json.dump({"features": FEATURES, "best_iteration": model.best_iteration,
                       "top_k": cfg.top_k, "score_floor": cfg.score_floor,
                       "no_match_threshold": cfg.no_match_threshold}, f, indent=2)
        vs1 = read_tsv(s1_path(cfg.data_dir, cfg.eval_split))["entity_id"].tolist()
        write_filtered(os.path.join(cfg.output_dir, f"{cfg.eval_split}_filtered_candidates.tsv"), fd, vs1)
        plot_pr(os.path.join(cfg.output_dir, "stage2_pr_curve.png"), ptab, s_val, y[is_ev],
                n_true_total, cfg.no_match_threshold)

    # ---- out-of-fold train_split candidates (Stage-3 training pairs) ------------------
    if cfg.oof_folds > 1:
        with stage(f"out-of-fold filtered candidates for {cfg.train_split}"):
            s_oof = oof_scores(X, y, s1col, is_tr, m_fit, m_es, model, cfg)
            tr_det = det[is_tr].reset_index(drop=True)
            ofd = filter_candidates(tr_det, s_oof, cfg.top_k, cfg.score_floor, cfg.no_match_threshold)
            ts1 = read_tsv(s1_path(cfg.data_dir, cfg.train_split))["entity_id"].tolist()
            write_filtered(os.path.join(cfg.output_dir, f"{cfg.train_split}_filtered_candidates.tsv"),
                           ofd, ts1)
            # should match the val filter stats if OOF scores mimic the deployed model
            rep["oof_train_split"] = {"folds": cfg.oof_folds,
                                      "average_precision": round(average_precision(y[is_tr], s_oof), 4),
                                      **filter_stats(ofd, truth[cfg.train_split])}
            print(f"    OOF AP {rep['oof_train_split']['average_precision']} "
                  f"(val AP {rep['val_average_precision']}) | "
                  f"{rep['oof_train_split']['avg_pairs_per_s1']} pairs/S1 "
                  f"(val {rep['filter_chosen']['avg_pairs_per_s1']})")

    # ---- apply to test ---------------------------------------------------------------
    tdet_path = os.path.join(cfg.cache_dir, "test_candidates_detail.parquet")
    if cfg.apply_test and os.path.exists(tdet_path):
        with stage("apply to test"):
            del X
            trec, tdet, tembs = build_universe(cfg, ["test"], "test")
            TX = build_features(tdet, trec, load_vectorizers(cfg.cache_dir), tembs, cfg.n_jobs)
            ts = model.predict(TX, num_iteration=model.best_iteration)
            tfd = filter_candidates(tdet, ts, cfg.top_k, cfg.score_floor, cfg.no_match_threshold)
            ts1 = read_tsv(s1_path(cfg.data_dir, "test"))["entity_id"].tolist()
            out = os.path.join(cfg.output_dir, "filtered_candidates.tsv")
            write_filtered(out, tfd, ts1)
            kept = tfd[tfd["keep"]]
            ck = trec["ckey"].to_numpy()[tdet["q"].to_numpy()]
            rep["test"] = {"n_s1": len(ts1), "pairs_in": int(len(tdet)),
                           "pairs_passed": int(len(kept)),
                           "likely_no_match_s1": int(len(ts1) - kept["s1"].nunique()),
                           "mean_score_by_country": pd.Series(ts).groupby(ck).mean().round(4).to_dict(),
                           "output": out}
    elif cfg.apply_test:
        print(f"    (no {tdet_path}; skipping test filtering)")

    rep["runtime_s"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(cfg.output_dir, "stage2_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print_report(rep, imp)
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
    if "test" in rep:
        print(f"TEST: {rep['test']}")
    print(f"runtime {rep['runtime_s']}s\n" + "=" * 78, flush=True)


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(root, "Dataset", "student_resource", "dataset"))
    p.add_argument("--output-dir", default=os.path.join(root, "output"))
    p.add_argument("--cache-dir", default=os.path.join(root, "cache"))
    p.add_argument("--train-split", default="train_split")
    p.add_argument("--eval-split", default="val")
    p.add_argument("--n-jobs", type=int, default=default_n_jobs())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-s1", type=int, default=0, help="subsample training S1 (0 = all)")
    p.add_argument("--es-frac", type=float, default=0.15, help="grouped early-stopping holdout")
    # model
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--min-child-samples", type=int, default=50)
    p.add_argument("--num-rounds", type=int, default=3000)
    p.add_argument("--early-stopping", type=int, default=100)
    p.add_argument("--scale-pos-weight", type=float, default=1.0)
    p.add_argument("--ablation", action=argparse.BooleanOptionalAction, default=True)
    # filtering
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--score-floor", type=float, default=0.02)
    p.add_argument("--no-match-threshold", type=float, default=0.05)
    p.add_argument("--sweep-k", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10])
    p.add_argument("--sweep-floor", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--apply-test", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--oof-folds", type=int, default=0,
                   help="K>1: also write out-of-fold <train-split>_filtered_candidates.tsv "
                        "(Stage-3 training pairs)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
