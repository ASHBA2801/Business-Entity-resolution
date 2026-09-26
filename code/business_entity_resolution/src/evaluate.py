"""Blocking evaluation: candidate recall (pair-level, per source / country / channel),
entity-level full recall, and reduction ratio vs. brute-force all-pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import read_tsv


def _true_pairs(gt_path: str, q: pd.DataFrame, pools: dict) -> pd.DataFrame:
    gt = read_tsv(gt_path)
    gt = gt[gt["source1_entity_id"].isin(set(q["entity_id"]))]
    t = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    t = t[t["m"].fillna("") != ""]
    q_row = pd.Series(np.arange(len(q)), index=q["entity_id"].to_numpy())
    n_s2 = len(pools["S2"])
    p_row = pd.concat([
        pd.Series(np.arange(n_s2), index=pools["S2"]["entity_id"].to_numpy()),
        pd.Series(np.arange(len(pools["S3"])) + n_s2, index=pools["S3"]["entity_id"].to_numpy()),
    ])
    out = pd.DataFrame({
        "q": q_row.reindex(t["source1_entity_id"].to_numpy()).to_numpy(),
        "p": p_row.reindex(t["m"].to_numpy()).to_numpy(),
        "src": t["m"].str[:2].to_numpy(),
    })
    missing = int(out["p"].isna().sum())
    if missing:
        print(f"    WARNING: {missing} ground-truth IDs not found in the pool", flush=True)
    out["p"] = out["p"].fillna(-1).astype(np.int64)
    out["q"] = out["q"].astype(np.int64)
    out["country"] = q["ckey"].to_numpy()[out["q"].to_numpy()]
    return out


def _recall(hit: np.ndarray) -> float:
    return round(float(hit.mean()), 4) if len(hit) else float("nan")


def evaluate_candidates(pairs: pd.DataFrame, q: pd.DataFrame, pools: dict, gt_path: str,
                        channels: list[str], ks=(1, 5, 10, 20)) -> dict:
    truth = _true_pairs(gt_path, q, pools)
    n_pool = len(pools["S2"]) + len(pools["S3"])
    tkey = truth["q"].to_numpy() * n_pool + truth["p"].to_numpy()
    pkey = pairs["q"].to_numpy() * n_pool + pairs["p"].to_numpy()
    order = np.argsort(pkey)
    pkey_sorted = pkey[order]
    pos = np.searchsorted(pkey_sorted, tkey)
    pos_c = np.minimum(pos, len(pkey_sorted) - 1)
    found = (len(pkey_sorted) > 0) & (pkey_sorted[pos_c] == tkey) & (truth["p"].to_numpy() >= 0)
    match_row = np.where(found, order[pos_c], -1)
    truth["hit"] = found

    rep = {"n_true_pairs": int(len(truth)),
           "recall_union": _recall(found),
           "recall_by_source": {s: _recall(found[truth["src"] == s]) for s in ("S2", "S3")},
           "recall_by_country": {c: _recall(found[truth["country"] == c])
                                 for c in sorted(truth["country"].unique())}}

    # per channel (and per channel x source), plus recall@k curves to size K
    rep["recall_by_channel"] = {}
    for ch in channels:
        rank = np.zeros(len(truth), dtype=np.int32)
        rank[found] = pairs[f"{ch}_rank"].to_numpy()[match_row[found]]
        hit_ch = rank > 0
        rep["recall_by_channel"][ch] = {
            "all": _recall(hit_ch),
            **{s: _recall(hit_ch[truth["src"] == s]) for s in ("S2", "S3")},
            "recall_at_k": {k: _recall((rank > 0) & (rank <= k)) for k in ks},
        }
        if len(channels) > 1:
            others = np.zeros(len(truth), dtype=bool)
            for o in channels:
                if o != ch:
                    others |= pairs[f"{o}_rank"].to_numpy()[np.maximum(match_row, 0)] > 0
            others &= found
            rep["recall_by_channel"][ch]["unique_hits"] = int((hit_ch & ~others).sum())

    # entity-level: share of S1 entities (with >=1 true match) whose matches are ALL found
    ent = truth.groupby("q")["hit"].all()
    rep["entity_full_recall"] = _recall(ent.to_numpy())

    # reduction ratio vs brute force (all S1 x all S2/S3) and vs within-country brute force
    n_q = len(q)
    rep["reduction_ratio"] = round(1 - len(pairs) / max(n_q * n_pool, 1), 6)
    qc = q["ckey"].value_counts()
    pc = pd.concat([pools["S2"]["ckey"], pools["S3"]["ckey"]]).value_counts()
    within = float(sum(qc[c] * pc.get(c, 0) for c in qc.index))
    rep["reduction_ratio_within_country"] = round(1 - len(pairs) / max(within, 1), 6)
    cnt = pairs.groupby("q").size().reindex(np.arange(n_q), fill_value=0)
    p_src = np.where(pairs["p"].to_numpy() < len(pools["S2"]), "S2", "S3")
    rep["avg_candidates_by_source"] = {s: round(float((p_src == s).sum()) / max(n_q, 1), 2)
                                       for s in ("S2", "S3")}
    rep["candidates_per_s1_quantiles"] = {str(k): float(v) for k, v in
                                          cnt.quantile([0.0, 0.5, 0.9, 1.0]).items()}
    return rep


def print_report(r: dict) -> None:
    print("\n" + "=" * 72)
    print(f"BLOCKING SUMMARY  (queries = {r['queries']})")
    print("=" * 72)
    print(f"S1 entities processed      : {r['n_s1']:,}")
    print(f"Pool sizes                 : S2 {r['n_s2']:,} | S3 {r['n_s3']:,}")
    print(f"Candidate pairs            : {r['n_candidate_pairs']:,}")
    print(f"Avg candidates / S1        : {r['avg_candidates_per_s1']}")
    print(f"S1 with zero candidates    : {r['s1_with_no_candidates']:,}")
    e = r.get("eval")
    if e:
        print("-" * 72)
        print(f"CANDIDATE RECALL (ceiling) : {e['recall_union']:.4f}  over {e['n_true_pairs']:,} true pairs")
        print(f"  by source                : " + " | ".join(f"{k} {v:.4f}" for k, v in e["recall_by_source"].items()))
        print(f"  by country               : " + " | ".join(f"{k} {v:.4f}" for k, v in e["recall_by_country"].items()))
        for ch, v in e["recall_by_channel"].items():
            rk = " ".join(f"@{k}:{x:.3f}" for k, x in v["recall_at_k"].items())
            uh = f" | unique hits {v['unique_hits']:,}" if "unique_hits" in v else ""
            print(f"  {ch:<12} alone       : {v['all']:.4f} (S2 {v['S2']:.4f}, S3 {v['S3']:.4f}) | {rk}{uh}")
        print(f"Entity-level full recall   : {e['entity_full_recall']:.4f}")
        print(f"Reduction ratio            : {e['reduction_ratio']:.6f} (all pairs) | "
              f"{e['reduction_ratio_within_country']:.6f} (within country)")
        print(f"Avg candidates by source   : {e['avg_candidates_by_source']}")
    print("-" * 72)
    print(f"Total runtime              : {r['total_runtime_s']}s")
    print(f"Peak memory (RSS)          : {r['peak_rss_gb']} GB")
    for s in r["stages"]:
        print(f"   {s['stage']:<38} {s['seconds']:>9.1f}s  peak {s['peak_rss_gb']} GB")
    print("=" * 72, flush=True)
