"""Stage 1: hybrid (sparse TF-IDF + dense BGE-M3) blocking -> candidate_pairs.tsv.

Usage (from code/business_entity_resolution/):
    python src/blocking_pipeline.py --data-dir <dataset> --queries val   # recall eval
    python src/blocking_pipeline.py --data-dir <dataset> --queries test  # submission file

--queries picks the Source-1 file and the pool it is matched against:
    val / train_split / train -> <split>_source1.tsv vs train/train_source{2,3}.tsv
    test                      -> test/test_source1.tsv vs test/test_source{2,3}.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import evaluate_candidates, print_report  # noqa: E402
from preprocessing import normalize_country, normalize_records  # noqa: E402
from sparse_retrieval import SparseBlocker  # noqa: E402
from utils import (STAGE_LOG, chunk_ranges, default_n_jobs, fork_map, get_shared,  # noqa: E402
                   peak_rss_gb, read_tsv, stage)

SOURCES = ("S2", "S3")
ALL_SOURCE_FILES = [f"{split}/{split}_source{i}.tsv" for split in ("train", "test")
                    for i in (1, 2, 3)]


def split_paths(data_dir: str, queries: str) -> dict:
    pool_split = "test" if queries == "test" else "train"
    gt = os.path.join(data_dir, queries, f"{queries}_ground_truth.tsv")
    return {
        "queries": os.path.join(data_dir, queries, f"{queries}_source1.tsv"),
        "S2": os.path.join(data_dir, pool_split, f"{pool_split}_source2.tsv"),
        "S3": os.path.join(data_dir, pool_split, f"{pool_split}_source3.tsv"),
        "gt": gt if os.path.exists(gt) else None,
    }


# --------------------------------------------------------------------------- loading
def _normalize_worker(rng):
    names, addrs, ctrs = get_shared("names"), get_shared("addrs"), get_shared("ctrs")
    s, e = rng
    return normalize_records(names[s:e], addrs[s:e], ctrs[s:e], get_shared("need_dense"))


def normalize_frame(df: pd.DataFrame, n_jobs: int, need_dense: bool) -> pd.DataFrame:
    names, addrs, ctrs = (df[c].tolist() for c in ("business_name", "business_address", "country"))
    parts = fork_map(_normalize_worker, chunk_ranges(len(df), 100_000), n_jobs,
                     shared={"names": names, "addrs": addrs, "ctrs": ctrs,
                             "need_dense": need_dense})
    arrow = lambda i: pd.array([t for p in parts for t in p[i]], dtype="string[pyarrow]")  # noqa: E731
    out = pd.DataFrame({
        "entity_id": df["entity_id"].to_numpy(),
        "ckey": np.array([normalize_country(c) for c in ctrs], dtype=object),
        "sparse_text": arrow(2),
        "name_text": arrow(4),
    })
    if need_dense:
        out["dense_text"] = arrow(3)
    return out


def partition_rows(ckeys: np.ndarray, country: str) -> np.ndarray:
    """Rows in a country partition. Records with an empty country label are treated as
    wildcards and joined to every partition (none exist in the provided data)."""
    return np.flatnonzero((ckeys == country) | (ckeys == ""))


# --------------------------------------------------------------------------- sparse
# channel name -> (text column, cfg attribute holding its top-K per source)
SPARSE_CHANNELS = {"sparse": ("sparse_text", "k_sparse"),       # name + address
                   "sparse_name": ("name_text", "k_sparse_name")}  # name only


def load_or_fit_vectorizers(cfg, countries: list[str]) -> dict:
    """One TF-IDF vocabulary/IDF per (text column, country), fitted unsupervised (no
    labels) on a random sample of ALL source records (train + test, S1/S2/S3)."""
    path = os.path.join(cfg.cache_dir, "sparse_vectorizers.pkl")
    cols = [col for col, _ in SPARSE_CHANNELS.values()]
    cache = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("_signature") != _sparse_signature(cfg):
            cache = {}
    missing = [(col, c) for col in cols for c in countries if (col, c) not in cache]
    if not missing:
        return cache
    texts: dict[tuple, list[str]] = {}
    for rel in ALL_SOURCE_FILES:
        p = os.path.join(cfg.data_dir, rel)
        if not os.path.exists(p):
            continue
        df = read_tsv(p)
        df = df.sample(n=min(len(df), cfg.fit_sample_per_file), random_state=0)
        norm = normalize_frame(df, cfg.n_jobs, need_dense=False)
        for c, grp in norm.groupby("ckey"):
            for col in cols:
                texts.setdefault((col, c), []).extend(grp[col].tolist())
        print(f"    fit corpus: +{len(df):,} rows from {rel}", flush=True)
    for col, c in missing:
        docs = texts.get((col, c)) or texts.get((col, ""), [])
        if not docs:  # country unseen everywhere: fit on all countries
            docs = [t for (cl, _), v in texts.items() if cl == col for t in v]
        blk = SparseBlocker(ngram_range=cfg.ngram_range, min_df=cfg.min_df).fit(docs)
        cache[(col, c)] = blk.vectorizer
        print(f"    fitted TF-IDF [{col}] for '{c}': {len(docs):,} docs, "
              f"vocab {len(blk.vectorizer.vocabulary_):,}", flush=True)
    cache["_signature"] = _sparse_signature(cfg)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    return cache


def _sparse_signature(cfg) -> str:
    return (f"{os.path.abspath(cfg.data_dir)}|{cfg.ngram_range}|{cfg.min_df}|"
            f"{cfg.fit_sample_per_file}|{sorted(SPARSE_CHANNELS.items())}")


def run_sparse(cfg, q: pd.DataFrame, pools: dict, countries: list[str], channel: str,
               vecs: dict) -> list[tuple]:
    col, k_attr = SPARSE_CHANNELS[channel]
    out = []
    for c in countries:
        blk = SparseBlocker(top_features=cfg.top_features, max_df_frac=cfg.max_df_frac,
                            k=getattr(cfg, k_attr), n_jobs=cfg.n_jobs,
                            max_work_per_chunk=cfg.max_work_per_chunk)
        blk.vectorizer = vecs[(col, c)]
        qi = partition_rows(q["ckey"].to_numpy(), c)
        t0 = time.perf_counter()
        Q = blk.transform(q[col].iloc[qi].tolist())
        for src in SOURCES:
            pool = pools[src]
            pi = partition_rows(pool["ckey"].to_numpy(), c)
            C = blk.transform(pool[col].iloc[pi].tolist())
            CT = blk.build_index(C)
            idx, sc = blk.search(Q, CT)
            out.append(_pairs(qi, pi, idx, sc, src))
            print(f"    {channel} [{c}/{src}] queries {len(qi):,} x pool {len(pi):,} | "
                  f"index nnz {CT.nnz:,} | work {blk.last_total_work:.3g} | "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
            del C, CT
    return out


# --------------------------------------------------------------------------- dense
def run_dense(cfg, q: pd.DataFrame, pools: dict, countries: list[str], tag: str) -> list[tuple]:
    from dense_retrieval import DenseBlocker  # lazy: imports torch
    blk = DenseBlocker(model_name=cfg.dense_model, batch_size=cfg.dense_batch_size,
                       max_seq_length=cfg.dense_max_len, k=cfg.k_dense,
                       n_threads=cfg.n_jobs, nprobe=cfg.nprobe)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    emb = {}
    for name, df, split in [("S1", q, tag)] + [(s, pools[s], cfg.pool_split) for s in SOURCES]:
        t0 = time.perf_counter()
        emb[name] = blk.encode(df["dense_text"].tolist(),
                               cache_path=os.path.join(cfg.cache_dir, f"dense_{split}_{name}.npy"))
        print(f"    encoded {name} ({split}): {len(df):,} texts in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
    out = []
    for c in countries:
        qi = partition_rows(q["ckey"].to_numpy(), c)
        qe = np.asarray(emb["S1"][qi], dtype=np.float32)
        for src in SOURCES:
            t0 = time.perf_counter()
            pi = partition_rows(pools[src]["ckey"].to_numpy(), c)
            index = blk.build_index(emb[src][pi])
            idx, sc = blk.search(index, qe)
            out.append(_pairs(qi, pi, idx, sc, src))
            print(f"    dense [{c}/{src}] queries {len(qi):,} x pool {len(pi):,} | "
                  f"{blk.last_index_type} | {time.perf_counter() - t0:.1f}s", flush=True)
    return out


# --------------------------------------------------------------------------- combine
def _pairs(qi, pi, idx, sc, src):
    """Map partition-local top-k results back to global rows -> (q_row, src, pool_row,
    score, rank) arrays, dropping -1 padding."""
    k = idx.shape[1]
    valid = idx >= 0
    q_rows = np.repeat(qi, k).reshape(-1, k)[valid]
    p_rows = pi[idx[valid]]
    ranks = np.tile(np.arange(1, k + 1, dtype=np.int16), (len(qi), 1))[valid]
    return q_rows, src, p_rows, sc[valid], ranks


def combine(channels: dict, n_s2: int) -> pd.DataFrame:
    """Union candidates across channels, one row per (query, candidate), keeping each
    channel's best score / rank (NaN / 0 when that channel did not retrieve it)."""
    frames = []
    for ch, parts in channels.items():
        for q_rows, src, p_rows, sc, rk in parts:
            gp = p_rows + (n_s2 if src == "S3" else 0)  # global pool row: S2 then S3
            frames.append(pd.DataFrame({"q": q_rows.astype(np.int64), "p": gp.astype(np.int64),
                                        f"{ch}_score": sc, f"{ch}_rank": rk}))
    df = pd.concat(frames, ignore_index=True)
    agg = {c: ("max" if c.endswith("_score") else "min") for c in df.columns if c not in ("q", "p")}
    df = df.groupby(["q", "p"], sort=True, as_index=False).agg(agg)
    for c in df.columns:
        if c.endswith("_rank"):
            df[c] = df[c].fillna(0).astype(np.int16)
    return df


def write_candidates(path: str, q_ids: np.ndarray, pool_ids: np.ndarray, pairs: pd.DataFrame):
    """One row per Source-1 entity (empty list if no candidates), comma-joined IDs."""
    qv, pv = pairs["q"].to_numpy(), pairs["p"].to_numpy()
    bounds = np.searchsorted(qv, np.arange(len(q_ids) + 1))
    cand_ids = pool_ids[pv]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, qid in enumerate(q_ids):
            f.write(f"{qid}\t{','.join(cand_ids[bounds[i]:bounds[i + 1]])}\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- main
def run(cfg) -> dict:
    t_start = time.perf_counter()
    paths = split_paths(cfg.data_dir, cfg.queries)
    cfg.pool_split = "test" if cfg.queries == "test" else "train"
    use_dense = not cfg.no_dense
    use_sparse = not cfg.no_sparse

    with stage("load + normalize"):
        q = normalize_frame(read_tsv(paths["queries"]), cfg.n_jobs, use_dense)
        pools = {s: normalize_frame(read_tsv(paths[s]), cfg.n_jobs, use_dense) for s in SOURCES}
        countries = sorted(c for c in set(q["ckey"]) if c) or [""]
        print(f"    queries {len(q):,} | S2 {len(pools['S2']):,} | S3 {len(pools['S3']):,} | "
              f"countries {countries}", flush=True)

    channels = {}
    if use_sparse:
        with stage("fit/load TF-IDF vectorizers"):
            vecs = load_or_fit_vectorizers(cfg, countries)
        for ch in SPARSE_CHANNELS:
            if ch == "sparse_name" and cfg.k_sparse_name <= 0:
                continue
            with stage(f"{ch} retrieval (TF-IDF char {cfg.ngram_range})"):
                channels[ch] = run_sparse(cfg, q, pools, countries, ch, vecs)
    if use_dense:
        with stage("dense retrieval (BGE-M3 + FAISS)"):
            channels["dense"] = run_dense(cfg, q, pools, countries, tag=cfg.queries)

    with stage("combine + write"):
        n_s2 = len(pools["S2"])
        pairs = combine(channels, n_s2)
        pool_ids = np.concatenate([pools["S2"]["entity_id"].to_numpy(),
                                   pools["S3"]["entity_id"].to_numpy()]).astype(object)
        q_ids = q["entity_id"].to_numpy().astype(object)
        os.makedirs(cfg.output_dir, exist_ok=True)
        name = "candidate_pairs.tsv" if cfg.queries == "test" else f"{cfg.queries}_candidate_pairs.tsv"
        out_path = os.path.join(cfg.output_dir, name)
        write_candidates(out_path, q_ids, pool_ids, pairs)
        if cfg.save_detail:
            detail = pairs.assign(source1_entity_id=q_ids[pairs["q"].to_numpy()],
                                  candidate_entity_id=pool_ids[pairs["p"].to_numpy()])
            detail.drop(columns=["q", "p"]).to_parquet(
                os.path.join(cfg.cache_dir, f"{cfg.queries}_candidates_detail.parquet"),
                index=False)
        print(f"    wrote {out_path}", flush=True)

    report = {"queries": cfg.queries, "n_s1": len(q), "n_s2": n_s2, "n_s3": len(pools["S3"]),
              "n_candidate_pairs": int(len(pairs)),
              "avg_candidates_per_s1": round(len(pairs) / max(len(q), 1), 2),
              "s1_with_no_candidates": int(len(q) - pairs["q"].nunique()),
              "output": out_path}
    if paths["gt"]:
        with stage("evaluate recall"):
            report["eval"] = evaluate_candidates(
                pairs=pairs, q=q, pools=pools, gt_path=paths["gt"],
                channels=list(channels))
    report["total_runtime_s"] = round(time.perf_counter() - t_start, 1)
    report["peak_rss_gb"] = round(peak_rss_gb(), 2)
    report["stages"] = STAGE_LOG
    report["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()}
    with open(os.path.join(cfg.output_dir, f"blocking_report_{cfg.queries}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print_report(report)
    return report


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(root, "Dataset", "student_resource", "dataset"))
    p.add_argument("--queries", default="val", choices=["val", "train_split", "train", "test"])
    p.add_argument("--output-dir", default=os.path.join(root, "output"))
    p.add_argument("--cache-dir", default=os.path.join(root, "cache"))
    p.add_argument("--n-jobs", type=int, default=default_n_jobs())
    # sparse
    p.add_argument("--k-sparse", type=int, default=20, help="top-K per source (S2 and S3)")
    p.add_argument("--k-sparse-name", type=int, default=10,
                   help="top-K per source for the name-only TF-IDF channel (0 disables)")
    p.add_argument("--ngram-range", type=int, nargs=2, default=(2, 4))
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--top-features", type=int, default=48,
                   help="n-grams kept per record (token-blocking prune)")
    p.add_argument("--max-df-frac", type=float, default=0.02,
                   help="drop n-grams whose posting list exceeds this fraction of the pool")
    p.add_argument("--fit-sample-per-file", type=int, default=150_000,
                   help="rows sampled from each of the 6 source files to fit TF-IDF")
    p.add_argument("--max-work-per-chunk", type=float, default=10e6,
                   help="posting-list work per search chunk (bounds per-worker memory)")
    # dense
    p.add_argument("--k-dense", type=int, default=20, help="top-K per source (S2 and S3)")
    p.add_argument("--dense-model", default="BAAI/bge-m3")
    p.add_argument("--dense-batch-size", type=int, default=64)
    p.add_argument("--dense-max-len", type=int, default=64)
    p.add_argument("--nprobe", type=int, default=32)
    # switches
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--no-sparse", action="store_true")
    p.add_argument("--save-detail", action=argparse.BooleanOptionalAction, default=True,
                   help="save per-candidate channel scores/ranks parquet (Stage-2 features)")
    cfg = p.parse_args(argv)
    cfg.ngram_range = tuple(cfg.ngram_range)
    return cfg


if __name__ == "__main__":
    run(parse_args())
