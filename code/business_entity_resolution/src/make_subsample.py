"""Build a small, relationship-preserving prototype dataset with the same layout as the
real one, so blocking_pipeline.py runs on it unchanged (--data-dir <out-dir>).

  val/          n_val S1 entities from val (stratified by country x singleton), + GT
  train_split/  n_train S1 entities from train_split, + GT (their matches act as
                realistic "other cluster" distractors in the pool)
  train/        pool = all matches of the above + n_unmatched never-matched records
                per source; train_source1/ground_truth = val + train_split selection
  test/         n_test_s1 test S1 + n_test_pool random test S2/S3 records (no labels;
                exercises the unseen-country path and the output validator)
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import read_tsv  # noqa: E402


def _write(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _stratified(s1: pd.DataFrame, gt: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    d = s1.merge(gt, left_on="entity_id", right_on="source1_entity_id")
    d["_single"] = d["matched_entity_ids"] == ""
    frac = min(1.0, n / len(d))
    return (d.groupby(["country", "_single"], group_keys=False)
             .sample(frac=frac, random_state=seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-unmatched", type=int, default=3500)
    ap.add_argument("--n-test-s1", type=int, default=500)
    ap.add_argument("--n-test-pool", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    D, O = a.data_dir, a.out_dir

    val = _stratified(read_tsv(f"{D}/val/val_source1.tsv"),
                      read_tsv(f"{D}/val/val_ground_truth.tsv"), a.n_val, a.seed)
    trn = _stratified(read_tsv(f"{D}/train_split/train_split_source1.tsv"),
                      read_tsv(f"{D}/train_split/train_split_ground_truth.tsv"), a.n_train, a.seed)
    s1_cols = ["entity_id", "business_name", "business_address", "country"]
    gt_cols = ["source1_entity_id", "matched_entity_ids"]
    _write(val[s1_cols], f"{O}/val/val_source1.tsv")
    _write(val[gt_cols], f"{O}/val/val_ground_truth.tsv")
    _write(trn[s1_cols], f"{O}/train_split/train_split_source1.tsv")
    _write(trn[gt_cols], f"{O}/train_split/train_split_ground_truth.tsv")
    both = pd.concat([val, trn])
    _write(both[s1_cols], f"{O}/train/train_source1.tsv")
    _write(both[gt_cols], f"{O}/train/train_ground_truth.tsv")

    wanted = {i for m in both["matched_entity_ids"] for i in m.split(",") if i}
    all_matched = {i for m in read_tsv(f"{D}/train/train_ground_truth.tsv")["matched_entity_ids"]
                   for i in m.split(",") if i}
    for src in ("2", "3"):
        pool = read_tsv(f"{D}/train/train_source{src}.tsv")
        keep = pool[pool["entity_id"].isin(wanted)]
        unmatched = pool[~pool["entity_id"].isin(all_matched)]
        extra = unmatched.sample(n=min(a.n_unmatched, len(unmatched)), random_state=a.seed)
        out = pd.concat([keep, extra]).sample(frac=1.0, random_state=a.seed)
        _write(out, f"{O}/train/train_source{src}.tsv")
        print(f"train S{src}: {len(keep):,} matched + {len(extra):,} unmatched = {len(out):,}")
        del pool

    for src, n in (("1", a.n_test_s1), ("2", a.n_test_pool), ("3", a.n_test_pool)):
        t = read_tsv(f"{D}/test/test_source{src}.tsv")
        t = t.groupby("country", group_keys=False).sample(frac=min(1.0, n / len(t)),
                                                          random_state=a.seed)
        _write(t, f"{O}/test/test_source{src}.tsv")
        print(f"test S{src}: {len(t):,} {t['country'].value_counts().to_dict()}")
    print(f"val S1: {len(val):,} {val['country'].value_counts().to_dict()} | "
          f"train_split S1: {len(trn):,}")


if __name__ == "__main__":
    main()
