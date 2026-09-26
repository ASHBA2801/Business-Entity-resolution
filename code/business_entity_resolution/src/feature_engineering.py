"""Pairwise features for (Source-1 entity, candidate) pairs — Stage 2 (and reusable by 3/4).

Inputs are the Stage-1 artifacts (explicit paths, passed in by stage2_lgbm.py), reused
rather than recomputed:
  * candidates detail parquet   per-pair channel scores / ranks
  * vectorizers pickle          per-country char TF-IDF (unsupervised)
  * embedding .npy (+ .json)    BGE-M3 embeddings per source file (L2-normalized, fp16)
    -> `bge_m3_cos` ("Feature 58"): the cosine of the cached embeddings, computed for
       EVERY candidate pair (not only the ones the dense channel retrieved).

Feature groups (see FEATURE_GROUPS):
  stage1     channel scores/ranks from blocking
  name       string similarity on normalized names, legal-suffix-stripped "core" names,
             abbreviation-aware coverage, acronym match
  address    string similarity, token/component overlap, numeric-token agreement
  tfidf      full (unpruned) char TF-IDF cosine with Stage-1 vectorizers
  dense      bge_m3_cos
  structure  competition context within the S1's candidate list and across S1s that
             retrieved the same candidate

No country is hard-coded: country only enters through the same-country flag and the
per-country TF-IDF vectorizer lookup (an unseen label falls back gracefully).
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from preprocessing import normalize_address, normalize_country, normalize_records
from utils import chunk_ranges, fork_map, get_shared, read_tsv

# Canonical legal-form tokens (post-normalization: "pvt ltd" -> "private limited").
# Generic across jurisdictions; includes common French/European forms because the
# test set has a country unseen in training.
LEGAL_TOKENS = frozenset({
    "corporation", "incorporated", "limited", "private", "company", "llc", "llp", "lp",
    "pllc", "plc", "pc", "pa", "ltda", "gmbh", "ag", "bv", "nv", "sa", "sas", "sasu",
    "sarl", "eurl", "sci", "snc", "scop", "dba", "ta", "the", "opc",
})
ACRONYM_STOP = frozenset({"and", "of", "the", "for", "a", "an", "de", "des", "du", "la", "le", "et"})
_DIGITS = re.compile(r"\d+")

FEATURE_GROUPS = {
    "stage1": ["sparse_score", "sparse_rank", "sparse_name_score", "sparse_name_rank",
               "dense_rank", "n_channels", "cand_is_s3", "same_country"],
    "name": ["name_lev", "name_jw", "name_tsort", "name_tset", "name_partial", "name_jacc",
             "name_wjacc", "name_lcsub", "name_exact", "core_lev", "core_tset", "core_jacc",
             "core_exact", "core_contained", "legal_match", "name_abbrev_sim",
             "acronym_match", "name_first_tok", "name_ntok_diff", "name_len_ratio",
             "name_num_match"],
    "address": ["addr_lev", "addr_tsort", "addr_tset", "addr_partial", "addr_jacc",
                "addr_wjacc", "addr_contain", "addr_lcsub", "addr_num_jacc",
                "addr_num_overlap", "addr_num_conflict", "addr_first_num", "addr_last_num",
                "addr_comp_jacc", "addr_tail_jacc", "addr_missing_s1", "addr_missing_cand",
                "addr_len_ratio"],
    "tfidf": ["full_tfidf_cos", "name_tfidf_cos"],
    "dense": ["bge_m3_cos"],
    "structure": ["grp_n_cands", "grp_n_same_src", "grp_rank_tfidf", "grp_gap_tfidf",
                  "grp_rank_bge", "grp_gap_bge", "grp_rank_name", "grp_gap_name",
                  "grp_margin_tfidf", "sim_to_top1", "cand_n_s1", "cand_rank_s1",
                  "cand_gap_s1"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
DENSE_DERIVED = {"bge_m3_cos", "dense_rank", "grp_rank_bge", "grp_gap_bge"}


# =========================================================================== records
def _norm_worker(rng):
    n, a, c = get_shared("n"), get_shared("a"), get_shared("c")
    s, e = rng
    nn, na, st, _, sn = normalize_records(n[s:e], a[s:e], c[s:e], need_dense=False)
    comps = [_addr_components(x, y) for x, y in zip(a[s:e], c[s:e])]
    return nn, na, st, sn, comps


def load_records(files: dict[str, str], n_jobs: int) -> pd.DataFrame:
    """Concatenate source files -> one frame indexed by entity_id with raw + normalized
    fields and each record's row in its own file (= its row in the embedding cache).

    files: {tag: path}, tag in {"S1", "S2", "S3"} (S1 may list several splits via
    tags like "S1:val"; the part after ':' names the embedding cache split)."""
    frames = []
    for tag, path in files.items():
        df = read_tsv(path)
        cols = [df[c].tolist() for c in ("business_name", "business_address", "country")]
        parts = fork_map(_norm_worker, chunk_ranges(len(df), 50_000), n_jobs,
                         shared={"n": cols[0], "a": cols[1], "c": cols[2]})
        flat = lambda i: [t for p in parts for t in p[i]]  # noqa: E731
        frames.append(pd.DataFrame({
            "entity_id": df["entity_id"].to_numpy(),
            "ckey": [normalize_country(c) for c in cols[2]],
            "name": flat(0), "addr": flat(1), "sparse_text": flat(2), "name_text": flat(3),
            "comps": flat(4),
            "emb_tag": tag, "emb_row": np.arange(len(df), dtype=np.int64),
        }))
    rec = pd.concat(frames, ignore_index=True)
    rec = rec.drop_duplicates("entity_id").set_index("entity_id")
    return rec


# =========================================================================== helpers
def _toks(s: str) -> list[str]:
    return s.split() if s else []


def core_name(name: str) -> str:
    """Normalized name with legal-form tokens removed (falls back to the full name if
    stripping would leave nothing, e.g. 'The Company')."""
    t = [w for w in name.split() if w not in LEGAL_TOKENS]
    return " ".join(t) if t else name


def _jacc(a: set, b: set) -> float:
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


def _wjacc(a: set, b: set, idf: dict, default: float) -> float:
    if not a and not b:
        return np.nan
    u = sum(idf.get(t, default) for t in a | b)
    return sum(idf.get(t, default) for t in a & b) / u if u else np.nan


def _lcsub_ratio(a: str, b: str) -> float:
    """Longest common substring length / length of the shorter string."""
    if not a or not b:
        return np.nan
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size / min(len(a), len(b))


def _acronyms(tokens: list[str]) -> set[str]:
    out = set()
    if len(tokens) >= 2:
        out.add("".join(t[0] for t in tokens))
    kept = [t for t in tokens if t not in ACRONYM_STOP]
    if len(kept) >= 2:
        out.add("".join(t[0] for t in kept))
    return out


def acronym_match(t1: list[str], t2: list[str]) -> float:
    """1 if one name contains (or equals, joined) the initialism of the other."""
    a1, a2 = _acronyms(t1), _acronyms(t2)
    j1, j2 = "".join(t1), "".join(t2)
    s1, s2 = set(t1), set(t2)
    hit = bool(a2 & (s1 | {j1})) or bool(a1 & (s2 | {j2}))
    return float(hit)


def abbrev_sim(t1: list[str], t2: list[str]) -> float:
    """Share of tokens in the shorter name that match a token of the other exactly, as a
    prefix abbreviation ('tech' ~ 'technologies', 'mgmt'-style handled upstream), or as
    an initialism of consecutive tokens ('ibm' ~ 'international business machines')."""
    if not t1 or not t2:
        return np.nan
    a, b = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
    bset = set(b)
    hit = 0
    for t in a:
        if t in bset:
            hit += 1
            continue
        ok = False
        if len(t) >= 2:
            for u in b:
                if (len(u) >= 2 and (u.startswith(t) or t.startswith(u))):
                    ok = True
                    break
            if not ok and len(t) <= len(b) and t.isalpha():
                for i in range(len(b) - len(t) + 1):
                    if "".join(w[0] for w in b[i:i + len(t)]) == t:
                        ok = True
                        break
        hit += ok
    return hit / len(a)


def _nums(s: str) -> list[str]:
    return _DIGITS.findall(s) if s else []


def _match_or_nan(x, y) -> float:
    if x is None or y is None:
        return np.nan
    return float(x == y)


def _addr_components(raw: str, country) -> list[str]:
    """Comma-separated address components, each normalized (placeholders dropped)."""
    if not raw:
        return []
    comps = [normalize_address(p, country) for p in str(raw).split(",")]
    return [c for c in comps if c]


def token_idf(texts, min_df: int = 1) -> tuple[dict, float]:
    """Unsupervised token IDF (smoothed) over a text collection."""
    df = Counter()
    n = 0
    for t in texts:
        n += 1
        df.update(set(t.split()))
    idf = {k: math.log((1 + n) / (1 + v)) + 1 for k, v in df.items() if v >= min_df}
    return idf, math.log(1 + n) + 1


# =========================================================================== per-pair
def _string_worker(rng):
    """String features for pair rows [s, e) — pure CPU, run in forked workers."""
    s, e = rng
    qi, pi = get_shared("qi")[s:e], get_shared("pi")[s:e]
    names, addrs, cores = get_shared("names"), get_shared("addrs"), get_shared("cores")
    comps, ckeys = get_shared("comps"), get_shared("ckeys")
    nidf, nidf0, aidf, aidf0 = get_shared("nidf"), get_shared("nidf0"), get_shared("aidf"), get_shared("aidf0")

    n1 = [names[i] for i in qi]; n2 = [names[i] for i in pi]
    a1 = [addrs[i] for i in qi]; a2 = [addrs[i] for i in pi]
    c1 = [cores[i] for i in qi]; c2 = [cores[i] for i in pi]
    f = {}
    kw = dict(workers=1, dtype=np.float32)
    f["name_lev"] = cpdist(n1, n2, scorer=Levenshtein.normalized_similarity, **kw)
    f["name_jw"] = cpdist(n1, n2, scorer=JaroWinkler.similarity, **kw)
    f["name_tsort"] = cpdist(n1, n2, scorer=fuzz.token_sort_ratio, **kw) / 100
    f["name_tset"] = cpdist(n1, n2, scorer=fuzz.token_set_ratio, **kw) / 100
    f["name_partial"] = cpdist(n1, n2, scorer=fuzz.partial_ratio, **kw) / 100
    f["core_lev"] = cpdist(c1, c2, scorer=Levenshtein.normalized_similarity, **kw)
    f["core_tset"] = cpdist(c1, c2, scorer=fuzz.token_set_ratio, **kw) / 100
    f["addr_lev"] = cpdist(a1, a2, scorer=Levenshtein.normalized_similarity, **kw)
    f["addr_tsort"] = cpdist(a1, a2, scorer=fuzz.token_sort_ratio, **kw) / 100
    f["addr_tset"] = cpdist(a1, a2, scorer=fuzz.token_set_ratio, **kw) / 100
    f["addr_partial"] = cpdist(a1, a2, scorer=fuzz.partial_ratio, **kw) / 100

    loop_cols = [c for g in ("name", "address") for c in FEATURE_GROUPS[g] if c not in f]
    out = {c: np.full(e - s, np.nan, dtype=np.float32) for c in loop_cols}
    for k in range(e - s):
        x1, x2, y1, y2, z1, z2 = n1[k], n2[k], a1[k], a2[k], c1[k], c2[k]
        tn1, tn2 = _toks(x1), _toks(x2)
        sn1, sn2 = set(tn1), set(tn2)
        tc1, tc2 = _toks(z1), _toks(z2)
        ta1, ta2 = set(_toks(y1)), set(_toks(y2))
        # name
        out["name_jacc"][k] = _jacc(sn1, sn2)
        out["name_wjacc"][k] = _wjacc(sn1, sn2, nidf, nidf0)
        out["name_lcsub"][k] = _lcsub_ratio(x1, x2)
        out["name_exact"][k] = float(x1 == x2 and x1 != "")
        out["core_jacc"][k] = _jacc(set(tc1), set(tc2))
        out["core_exact"][k] = float(z1 == z2 and z1 != "")
        out["core_contained"][k] = float(bool(z1) and bool(z2) and (z1 in z2 or z2 in z1))
        l1, l2 = sn1 & LEGAL_TOKENS, sn2 & LEGAL_TOKENS
        out["legal_match"][k] = float(l1 == l2) if (l1 and l2) else np.nan
        out["name_abbrev_sim"][k] = abbrev_sim(tc1, tc2)
        out["acronym_match"][k] = acronym_match(tc1, tc2)
        out["name_first_tok"][k] = float(bool(tc1) and bool(tc2) and tc1[0] == tc2[0])
        out["name_ntok_diff"][k] = abs(len(tn1) - len(tn2))
        out["name_len_ratio"][k] = (min(len(x1), len(x2)) / max(len(x1), len(x2))
                                    if x1 and x2 else np.nan)
        nn1, nn2 = set(_nums(x1)), set(_nums(x2))
        out["name_num_match"][k] = float(bool(nn1 & nn2)) if (nn1 and nn2) else np.nan
        # address
        out["addr_jacc"][k] = _jacc(ta1, ta2)
        out["addr_wjacc"][k] = _wjacc(ta1, ta2, aidf, aidf0)
        out["addr_contain"][k] = (len(ta1 & ta2) / min(len(ta1), len(ta2))
                                  if ta1 and ta2 else np.nan)
        out["addr_lcsub"][k] = _lcsub_ratio(y1, y2)
        u1, u2 = _nums(y1), _nums(y2)
        if u1 and u2:
            s1, s2 = set(u1), set(u2)
            inter = len(s1 & s2)
            out["addr_num_jacc"][k] = inter / len(s1 | s2)
            out["addr_num_overlap"][k] = inter
            out["addr_num_conflict"][k] = float(inter == 0)
            out["addr_first_num"][k] = float(u1[0] == u2[0])
            # last long number: postcode-like (ZIP / PIN / code postal) — by shape, not country
            p1 = next((x for x in reversed(u1) if len(x) >= 4), None)
            p2 = next((x for x in reversed(u2) if len(x) >= 4), None)
            out["addr_last_num"][k] = _match_or_nan(p1, p2)
        q_c, p_c = comps[qi[k]], comps[pi[k]]
        if q_c and p_c:
            out["addr_comp_jacc"][k] = _jacc(set(q_c), set(p_c))
            t1 = set(" ".join(q_c[-2:]).split()); t2 = set(" ".join(p_c[-2:]).split())
            out["addr_tail_jacc"][k] = _jacc(t1, t2)
        out["addr_missing_s1"][k] = float(not y1)
        out["addr_missing_cand"][k] = float(not y2)
        out["addr_len_ratio"][k] = (min(len(y1), len(y2)) / max(len(y1), len(y2))
                                    if y1 and y2 else np.nan)
    f.update(out)
    return {c: np.asarray(v, dtype=np.float32) for c, v in f.items()}


# =========================================================================== vector sims
def rowwise_tfidf_cos(texts_q: list[str], texts_p: list[str], ckeys: np.ndarray,
                      vecs: dict, col: str, chunk: int = 200_000) -> np.ndarray:
    """Cosine of full (unpruned) TF-IDF vectors, pair by pair, per-country vectorizer.
    Unseen country (no fitted vectorizer) -> falls back to any fitted vectorizer."""
    out = np.full(len(texts_q), np.nan, dtype=np.float32)
    avail = [k for k in vecs if isinstance(k, tuple) and k[0] == col]
    for c in np.unique(ckeys):
        vec = vecs.get((col, c)) or (vecs[avail[0]] if avail else None)
        if vec is None:
            continue
        rows = np.flatnonzero(ckeys == c)
        for s in range(0, len(rows), chunk):
            r = rows[s:s + chunk]
            A = vec.transform([texts_q[i] for i in r])
            B = vec.transform([texts_p[i] for i in r])
            out[r] = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    return out


def _tfidf_transform_worker(rng):
    s, e = rng
    texts, idx = get_shared("texts"), get_shared("idx")
    return get_shared("vec").transform([texts[i] for i in idx[s:e]])


def _tfidf_cos_worker(rng):
    s, e = rng
    X, pl, pr = get_shared("X"), get_shared("pl"), get_shared("pr")
    return np.asarray(X[pl[s:e]].multiply(X[pr[s:e]]).sum(axis=1)).ravel()


def pairwise_tfidf_cos(texts: np.ndarray, left: np.ndarray, right: np.ndarray,
                       ckeys: np.ndarray, vecs: dict, col: str, n_jobs: int = 1,
                       chunk: int = 200_000) -> np.ndarray:
    """Same values as rowwise_tfidf_cos(texts[left], texts[right], ckeys, ...), but each
    distinct record is transformed once per vectorizer (records repeat across many pairs)
    and the transform and the row-wise products run in forked workers."""
    from scipy import sparse
    out = np.full(len(left), np.nan, dtype=np.float32)
    avail = [k for k in vecs if isinstance(k, tuple) and k[0] == col]
    for c in np.unique(ckeys):
        vec = vecs.get((col, c)) or (vecs[avail[0]] if avail else None)
        if vec is None:
            continue
        rows = np.flatnonzero(ckeys == c)
        idx = np.unique(np.concatenate([left[rows], right[rows]]))
        X = sparse.vstack(fork_map(_tfidf_transform_worker, chunk_ranges(len(idx), chunk), n_jobs,
                                   shared={"texts": texts, "idx": idx, "vec": vec}), format="csr")
        pl, pr = np.searchsorted(idx, left[rows]), np.searchsorted(idx, right[rows])
        parts = fork_map(_tfidf_cos_worker, chunk_ranges(len(rows), chunk), n_jobs,
                         shared={"X": X, "pl": pl, "pr": pr})
        out[rows] = np.concatenate(parts)
        del X
    return out


def load_embeddings(tag_to_path: dict[str, str | None],
                    expected_rows: dict[str, int]) -> dict:
    """Memmap Stage-1 BGE-M3 embedding files, one per source file (tag -> .npy path; the
    .npy.json sidecar written next to it records completeness). A file that is not given,
    missing, incomplete or of the wrong length is skipped (feature becomes NaN), never
    recomputed here."""
    embs = {}
    for tag, p in tag_to_path.items():
        if not p:
            print(f"    [dense] no embedding file for {tag} -> bge_m3_cos NaN")
            continue
        if not (os.path.exists(p) and os.path.exists(p + ".json")):
            print(f"    [dense] missing {p} (or its .json) -> bge_m3_cos NaN for {tag}")
            continue
        meta = json.load(open(p + ".json"))
        if meta.get("done") != meta.get("n") or meta.get("n") != expected_rows[tag]:
            print(f"    [dense] {p} incomplete/mismatched "
                  f"(done {meta.get('done')}/{meta.get('n')}, file rows {expected_rows[tag]})")
            continue
        embs[tag] = np.load(p, mmap_mode="r")
    return embs


def rowwise_dense_cos(rec: pd.DataFrame, qi: np.ndarray, pi: np.ndarray, embs: dict,
                      chunk: int = 100_000) -> np.ndarray:
    out = np.full(len(qi), np.nan, dtype=np.float32)
    if not embs:
        return out
    tags = rec["emb_tag"].to_numpy()
    erow = rec["emb_row"].to_numpy()
    qt, pt = tags[qi], tags[pi]
    for a in np.unique(qt):
        for b in np.unique(pt):
            if a not in embs or b not in embs:
                continue
            m = np.flatnonzero((qt == a) & (pt == b))
            for s in range(0, len(m), chunk):
                r = m[s:s + chunk]
                # sorted gathers keep memmap reads sequential-ish
                A = np.asarray(embs[a][erow[qi[r]]], dtype=np.float32)
                B = np.asarray(embs[b][erow[pi[r]]], dtype=np.float32)
                out[r] = np.einsum("ij,ij->i", A, B)
    return out


# =========================================================================== structure
def _group_rank_gap(df: pd.DataFrame, key: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    g = df.groupby(key, sort=False)[col]
    rank = g.rank(method="min", ascending=False).to_numpy(np.float32)
    gap = (g.transform("max") - df[col]).to_numpy(np.float32)
    return rank, gap


def _group_second_largest(keys: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Per row, the second-largest value in its key group (0.0 for one-row groups): the
    values of groupby(keys)[vals].transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1
    else 0.0), from one sort instead of a Python call per group."""
    order = np.lexsort((-vals, keys))
    k, v = keys[order], vals[order]
    first = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    size = np.diff(np.r_[first, len(k)])
    second = np.where(size > 1, v[np.minimum(first + 1, len(v) - 1)], 0.0)
    out = np.empty(len(vals), dtype=np.float64)
    out[order] = np.repeat(second, size)
    return out


def structural_features(pairs: pd.DataFrame, feats: pd.DataFrame, rec: pd.DataFrame,
                        vecs: dict, n_jobs: int = 1) -> pd.DataFrame:
    """Competition context. Within-S1: list size, rank / gap-to-best on key similarities,
    top1-top2 margin, and each candidate's similarity to the S1's top-1 candidate (a
    near-duplicate of the best candidate is likely another record of the same cluster).
    Across S1s: how many S1s retrieved this candidate, and this S1's rank / gap among
    them (computed over every S1 in `pairs` — pass the full query set, as at test)."""
    d = pd.DataFrame({"q": pairs["q"].to_numpy(), "p": pairs["p"].to_numpy(),
                      "src": pairs["cand_is_s3"].to_numpy(),
                      "t": feats["full_tfidf_cos"].fillna(0).to_numpy(),
                      "b": feats["bge_m3_cos"].to_numpy(),
                      "n": feats["name_tset"].to_numpy()})
    out = pd.DataFrame(index=feats.index)
    gq = d.groupby("q", sort=False)
    out["grp_n_cands"] = gq["p"].transform("size").to_numpy(np.float32)
    out["grp_n_same_src"] = d.groupby(["q", "src"], sort=False)["p"].transform("size").to_numpy(np.float32)
    out["grp_rank_tfidf"], out["grp_gap_tfidf"] = _group_rank_gap(d, "q", "t")
    if d["b"].notna().any():
        out["grp_rank_bge"], out["grp_gap_bge"] = _group_rank_gap(d, "q", "b")
    else:
        out["grp_rank_bge"] = out["grp_gap_bge"] = np.nan
    out["grp_rank_name"], out["grp_gap_name"] = _group_rank_gap(d, "q", "n")
    top2 = _group_second_largest(d["q"].to_numpy(), d["t"].to_numpy())
    out["grp_margin_tfidf"] = (gq["t"].transform("max").to_numpy() - top2).astype(np.float32)

    # similarity of each candidate to the S1's top-1 candidate (by full TF-IDF)
    top_idx = d.loc[d.groupby("q", sort=False)["t"].idxmax(), ["q", "p"]]
    top_p = d["q"].map(top_idx.set_index("q")["p"]).to_numpy()
    texts = rec["sparse_text"].to_numpy()
    ck = rec["ckey"].to_numpy()
    out["sim_to_top1"] = pairwise_tfidf_cos(texts, d["p"].to_numpy(), top_p,
                                            ck[d["p"].to_numpy()], vecs, "sparse_text", n_jobs)
    out.loc[d["p"].to_numpy() == top_p, "sim_to_top1"] = np.nan  # the top-1 itself

    gp = d.groupby("p", sort=False)
    out["cand_n_s1"] = gp["q"].transform("size").to_numpy(np.float32)
    out["cand_rank_s1"], out["cand_gap_s1"] = _group_rank_gap(d, "p", "t")
    return out


# =========================================================================== main entry
def build_features(pairs: pd.DataFrame, rec: pd.DataFrame, vecs: dict, embs: dict,
                   n_jobs: int, chunk: int = 20_000) -> pd.DataFrame:
    """pairs: columns q, p (integer rows into `rec`) + Stage-1 channel columns, sorted
    by q. Returns a float32 frame with FEATURES columns aligned to `pairs`."""
    qi = pairs["q"].to_numpy(np.int64)
    pi = pairs["p"].to_numpy(np.int64)
    names = rec["name"].tolist()
    cores = [core_name(n) for n in names]
    addrs = rec["addr"].tolist()
    ckeys = rec["ckey"].to_numpy()
    comps = rec["comps"].tolist()
    used = np.unique(np.concatenate([qi, pi]))
    nidf, nidf0 = token_idf(names[i] for i in used)
    aidf, aidf0 = token_idf(addrs[i] for i in used)

    parts = fork_map(_string_worker, chunk_ranges(len(pairs), chunk), n_jobs,
                     shared={"qi": qi, "pi": pi, "names": names, "addrs": addrs,
                             "cores": cores, "comps": comps, "ckeys": ckeys,
                             "nidf": nidf, "nidf0": nidf0, "aidf": aidf, "aidf0": aidf0})
    feats = pd.DataFrame({c: np.concatenate([p[c] for p in parts]) for c in parts[0]})

    # stage-1 channel features (dense_score dropped: bge_m3_cos supersedes it for all pairs)
    for c in ("sparse_score", "sparse_name_score"):
        feats[c] = pairs[c].to_numpy(np.float32) if c in pairs else np.nan
    for c in ("sparse_rank", "sparse_name_rank", "dense_rank"):
        v = pairs[c].to_numpy(np.float32) if c in pairs else np.zeros(len(pairs), np.float32)
        feats[c] = np.where(v > 0, v, np.nan)  # 0 = not retrieved by that channel
    rank_cols = [c for c in pairs.columns if c.endswith("_rank")]
    feats["n_channels"] = (pairs[rank_cols].to_numpy() > 0).sum(1).astype(np.float32)
    feats["cand_is_s3"] = pairs["cand_is_s3"].to_numpy(np.float32)
    feats["same_country"] = (ckeys[qi] == ckeys[pi]).astype(np.float32)

    st = rec["sparse_text"].to_numpy()
    nt = rec["name_text"].to_numpy()
    feats["full_tfidf_cos"] = pairwise_tfidf_cos(st, qi, pi, ckeys[qi], vecs, "sparse_text", n_jobs)
    feats["name_tfidf_cos"] = pairwise_tfidf_cos(nt, qi, pi, ckeys[qi], vecs, "name_text", n_jobs)
    feats["bge_m3_cos"] = rowwise_dense_cos(rec, qi, pi, embs)

    feats = pd.concat([feats, structural_features(pairs, feats, rec, vecs, n_jobs)], axis=1)
    return feats[FEATURES].astype(np.float32)


def load_vectorizers(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
