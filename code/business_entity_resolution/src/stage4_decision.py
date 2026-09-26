"""Stage 4: final match decision -> matching_results.tsv.

Turns Stage-3 pair scores into the set of matched S2/S3 ids for every Source-1 entity.
Decision rule, per S1: rank its scored candidates by `score_column` (best first) and keep
those with score >= `threshold`, at most `max_matches` of them (0 = no cap). An S1 with
nothing kept (or no scored pairs at all) is predicted as a singleton (empty list).

Two subcommands; every input and output is an explicit path (see README for the contract):

  tune    val pair scores + val S1 + val truth -> decision JSON (rule maximizing the
          challenge's macro F0.5 over every val S1), optional val matching TSV + report
  apply   pair scores + S1 + decision JSON (or --score-column/--threshold) -> matching TSV

    python src/stage4_decision.py tune --scores val_scores.tsv --s1 val_source1.tsv \\
           --truth val_ground_truth.tsv --output decision.json
    python src/stage4_decision.py apply --scores test_scores.tsv --s1 test_source1.tsv \\
           --decision decision.json --output matching_results.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import ensure_parent, load_truth, read_tsv, require_files  # noqa: E402

SCORE_COLUMNS = ("raw_score", "normalized_score")
THRESHOLDS = [round(float(t), 3) for t in np.arange(0.05, 0.99, 0.01)] + [0.99, 0.995]
CAPS = [1, 2, 3, 0]


# =========================================================================== data
def load_scores(path: str, all_s1: list[str]) -> pd.DataFrame:
    """Stage-3 scores TSV -> pairs with per-S1 rank for every score column."""
    d = pd.read_csv(path, sep="\t", dtype={"source1_entity_id": str, "candidate_entity_id": str},
                    keep_default_na=False)
    missing = [c for c in ("source1_entity_id", "candidate_entity_id", *SCORE_COLUMNS) if c not in d]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} (not a Stage-3 scores file?)")
    unknown = set(d["source1_entity_id"]) - set(all_s1)
    if unknown:
        raise ValueError(f"{path}: {len(unknown)} S1 ids not in the given --s1 file, "
                         f"e.g. {sorted(unknown)[:3]}")
    d = d.drop_duplicates(["source1_entity_id", "candidate_entity_id"])
    for c in SCORE_COLUMNS:
        d[c] = d[c].astype(np.float64)
        d[f"rank_{c}"] = d.groupby("source1_entity_id")[c].rank(method="first", ascending=False)
    return d.reset_index(drop=True)


def select(d: pd.DataFrame, col: str, threshold: float, max_matches: int) -> np.ndarray:
    keep = d[col].to_numpy() >= threshold
    if max_matches:
        keep &= d[f"rank_{col}"].to_numpy() <= max_matches
    return keep


# =========================================================================== metric
class MacroF05:
    """Challenge metric: mean per-S1 F0.5 over every S1 with ground truth; an S1 with no
    prediction scores 1.0 if it is a true singleton, else 0.0."""

    def __init__(self, d: pd.DataFrame, s1_ids: list[str], truth: dict):
        self.ids = [s for s in s1_ids if s in truth]
        if len(self.ids) < len(s1_ids):
            print(f"    WARNING: {len(s1_ids) - len(self.ids)} S1 ids have no ground truth "
                  "and are ignored by the metric", flush=True)
        code = pd.Series(np.arange(len(self.ids)), index=self.ids)
        c = code.reindex(d["source1_entity_id"].to_numpy())
        self.in_truth = c.notna().to_numpy()
        self.code = c.fillna(0).to_numpy(np.int64)
        self.n_true = np.array([len(truth[s]) for s in self.ids], dtype=np.float64)
        self.label = np.fromiter((b in truth.get(a, ()) for a, b in
                                  zip(d["source1_entity_id"], d["candidate_entity_id"])),
                                 bool, len(d))

    def __call__(self, keep: np.ndarray) -> float:
        keep = keep & self.in_truth
        E = len(self.ids)
        tp = np.bincount(self.code, weights=(keep & self.label).astype(np.float64), minlength=E)
        npred = np.bincount(self.code, weights=keep.astype(np.float64), minlength=E)
        with np.errstate(divide="ignore", invalid="ignore"):
            p, r = tp / npred, tp / self.n_true
            f = 1.25 * p * r / (0.25 * p + r)
        f = np.where(tp > 0, f, 0.0)
        f = np.where((npred == 0) & (self.n_true == 0), 1.0, f)
        return float(f.mean()) if E else float("nan")


# =========================================================================== output
def write_matches(path: str, d: pd.DataFrame, keep: np.ndarray, col: str, all_s1: list[str]):
    """One row per S1 in --s1 order; matched ids best-first, comma-joined (empty = none)."""
    k = d[keep].sort_values(["source1_entity_id", col], ascending=[True, False], kind="stable")
    by = k.groupby("source1_entity_id", sort=False)["candidate_entity_id"].agg(",".join).to_dict()
    tmp = ensure_parent(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in all_s1:
            f.write(f"{s}\t{by.get(s, '')}\n")
    os.replace(tmp, path)
    n_nonempty = sum(1 for s in all_s1 if s in by)
    print(f"    wrote {path}: {len(all_s1):,} S1, {n_nonempty:,} with >=1 match, "
          f"{int(keep.sum()):,} matched pairs", flush=True)


# =========================================================================== main
def run_tune(cfg):
    require_files(cfg.scores, cfg.s1, cfg.truth)
    all_s1 = read_tsv(cfg.s1)["entity_id"].tolist()
    d = load_scores(cfg.scores, all_s1)
    metric = MacroF05(d, all_s1, load_truth(cfg.truth))
    rows = [{"score_column": col, "threshold": t, "max_matches": cap,
             "macro_f05": round(metric(select(d, col, t, cap)), 5)}
            for col in SCORE_COLUMNS for cap in CAPS for t in THRESHOLDS]
    sweep = pd.DataFrame(rows)
    # best score; ties -> higher threshold, then tighter cap (more precise rule)
    sweep["_cap"] = sweep["max_matches"].replace(0, 10**6)
    sweep = sweep.sort_values(["macro_f05", "threshold", "_cap"], ascending=[False, False, True],
                              kind="stable").drop(columns="_cap").reset_index(drop=True)
    best = sweep.iloc[0]
    decision = {"score_column": best["score_column"], "threshold": float(best["threshold"]),
                "max_matches": int(best["max_matches"]),
                "tuned_macro_f05": float(best["macro_f05"]),
                "tuned_on": {"scores": os.path.basename(cfg.scores),
                             "s1": os.path.basename(cfg.s1), "n_s1": len(metric.ids)},
                "all_empty_macro_f05": round(metric(np.zeros(len(d), bool)), 5)}
    with open(ensure_parent(cfg.output), "w") as f:
        json.dump(decision, f, indent=2)
    print(f"    decision: {decision}\n    wrote {cfg.output}", flush=True)
    print("    top rules (val macro F0.5):\n" + sweep.head(10).to_string(index=False), flush=True)
    if cfg.sweep_output:
        sweep.to_csv(ensure_parent(cfg.sweep_output), sep="\t", index=False)
    if cfg.matches_output:
        keep = select(d, decision["score_column"], decision["threshold"], decision["max_matches"])
        write_matches(cfg.matches_output, d, keep, decision["score_column"], all_s1)
    return decision


def run_apply(cfg):
    require_files(cfg.scores, cfg.s1, cfg.decision, cfg.truth)
    if cfg.decision:
        dec = json.load(open(cfg.decision))
    elif cfg.threshold is not None:
        dec = {"score_column": cfg.score_column, "threshold": cfg.threshold,
               "max_matches": cfg.max_matches}
    else:
        raise SystemExit("apply: give --decision, or --threshold (with optional "
                         "--score-column / --max-matches)")
    all_s1 = read_tsv(cfg.s1)["entity_id"].tolist()
    d = load_scores(cfg.scores, all_s1)
    keep = select(d, dec["score_column"], dec["threshold"], dec["max_matches"])
    print(f"    rule: {dec['score_column']} >= {dec['threshold']}, max_matches "
          f"{dec['max_matches'] or 'unlimited'}", flush=True)
    write_matches(cfg.output, d, keep, dec["score_column"], all_s1)
    if cfg.truth:
        f05 = MacroF05(d, all_s1, load_truth(cfg.truth))(keep)
        print(f"    macro F0.5 vs {cfg.truth}: {f05:.4f}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tune", help="choose the decision rule on labelled scores (val)")
    i = t.add_argument_group("inputs")
    i.add_argument("--scores", required=True, help="Stage-3 pair scores TSV")
    i.add_argument("--s1", required=True, help="Source-1 TSV (every S1 is scored)")
    i.add_argument("--truth", required=True, help="ground truth for --s1")
    o = t.add_argument_group("outputs")
    o.add_argument("--output", required=True, help="decision JSON")
    o.add_argument("--matches-output", help="matching TSV for --s1 under the chosen rule")
    o.add_argument("--sweep-output", help="full rule sweep TSV")

    a = sub.add_parser("apply", help="write matching_results.tsv")
    i = a.add_argument_group("inputs")
    i.add_argument("--scores", required=True, help="Stage-3 pair scores TSV")
    i.add_argument("--s1", required=True, help="Source-1 TSV (defines the output rows)")
    i.add_argument("--decision", help="decision JSON from tune")
    i.add_argument("--truth", help="ground truth for --s1 (optional: prints macro F0.5)")
    a.add_argument("--score-column", choices=SCORE_COLUMNS, default="raw_score",
                   help="manual rule (used only without --decision)")
    a.add_argument("--threshold", type=float, help="manual rule (used only without --decision)")
    a.add_argument("--max-matches", type=int, default=0,
                   help="manual rule (used only without --decision); 0 = no cap")
    o = a.add_argument_group("outputs")
    o.add_argument("--output", required=True, help="matching TSV "
                                                   "(source1_entity_id, matched_entity_ids)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    (run_tune if args.cmd == "tune" else run_apply)(args)
