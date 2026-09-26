"""Sparse (character n-gram TF-IDF) blocking.

Scalability design (never materializes the |S1| x |pool| similarity matrix):
  (a) Exact blocking key: records are partitioned by normalized country label (lossless
      on train: 0 of 7.6M true pairs cross countries). Partitions are an open set.
  (b) Within a partition, similarity is only computed between S1 and pool records that
      share at least one *retained* n-gram:
  (c) Token blocking via pruning: each record keeps only its `top_features` highest
      TF-IDF-weighted n-grams (the inverted index is built on high-value tokens only),
      and n-grams whose posting list exceeds `max_df_frac` of the partition are removed
      from the index. Q_chunk @ C^T over these pruned vectors is exactly an inverted-index
      traversal, executed in C by scipy. Chunks are sized by estimated work (sum of
      posting-list lengths), so memory stays bounded at any corpus size.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import chunk_ranges, fork_map, get_shared


def prune_top_features(X: sp.csr_matrix, top_t: int) -> sp.csr_matrix:
    """Keep the `top_t` largest entries in each row, then re-L2-normalize rows."""
    X = X.tocsr()
    counts = np.diff(X.indptr)
    if top_t <= 0 or counts.max(initial=0) <= top_t:
        return X
    rows = np.repeat(np.arange(X.shape[0]), counts)
    order = np.lexsort((-X.data, rows))  # by row, then weight desc
    rank = np.arange(X.nnz) - X.indptr[rows[order]]
    keep = order[rank < top_t]
    keep.sort()  # keeps rows contiguous and column order within rows
    new_counts = np.minimum(counts, top_t)
    indptr = np.concatenate([[0], np.cumsum(new_counts)])
    Y = sp.csr_matrix((X.data[keep], X.indices[keep], indptr), shape=X.shape)
    norms = np.sqrt(np.asarray(Y.multiply(Y).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    Y.data /= np.repeat(norms, new_counts).astype(Y.data.dtype)
    return Y


def _topk_rows(R: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = R.shape[0]
    idx = np.full((n, k), -1, dtype=np.int64)
    sc = np.zeros((n, k), dtype=np.float32)
    indptr, indices, data = R.indptr, R.indices, R.data
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            continue
        d = data[s:e]
        p = np.argpartition(-d, k)[:k] if e - s > k else np.arange(e - s)
        p = p[np.argsort(-d[p], kind="stable")]
        idx[i, :len(p)] = indices[s:e][p]
        sc[i, :len(p)] = d[p]
    return idx, sc


def _transform_worker(rng):
    vec, texts, top_t = get_shared("vec"), get_shared("texts"), get_shared("top_t")
    X = vec.transform(texts[rng[0]:rng[1]])
    return prune_top_features(X.astype(np.float32), top_t)


def _search_worker(rng):
    Q, CT, k = get_shared("Q"), get_shared("CT"), get_shared("k")
    R = (Q[rng[0]:rng[1]] @ CT).tocsr()
    return _topk_rows(R, k)


class SparseBlocker:
    """Character n-gram TF-IDF retriever for one partition (country)."""

    def __init__(self, ngram_range=(2, 4), min_df=2, top_features=48, max_df_frac=0.02,
                 k=20, n_jobs=8, transform_chunk=50_000, max_work_per_chunk=30_000_000,
                 max_rows_per_chunk=4_000):
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.top_features = top_features
        self.max_df_frac = max_df_frac
        self.k = k
        self.n_jobs = n_jobs
        self.transform_chunk = transform_chunk
        self.max_work_per_chunk = max_work_per_chunk
        self.max_rows_per_chunk = max_rows_per_chunk
        self.vectorizer: TfidfVectorizer | None = None

    # -- vectorization -----------------------------------------------------------
    def fit(self, texts: list[str]) -> "SparseBlocker":
        """Unsupervised IDF fit (no labels). `texts` may be a sample of the partition."""
        min_df = self.min_df if len(texts) >= 50 * self.min_df else 1
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=self.ngram_range, min_df=min_df,
            sublinear_tf=True, lowercase=False, dtype=np.float32)
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> sp.csr_matrix:
        ranges = chunk_ranges(len(texts), self.transform_chunk)
        parts = fork_map(_transform_worker, ranges, self.n_jobs,
                         shared={"vec": self.vectorizer, "texts": texts,
                                 "top_t": self.top_features})
        if not parts:
            return sp.csr_matrix((0, len(self.vectorizer.vocabulary_)), dtype=np.float32)
        return sp.vstack(parts, format="csr")

    # -- retrieval ---------------------------------------------------------------
    def build_index(self, C: sp.csr_matrix) -> sp.csr_matrix:
        """Inverted index = C^T in CSR (feature -> postings), with over-long posting
        lists (n-grams present in > max_df_frac of the pool) dropped."""
        CT = C.T.tocsr()
        df = np.diff(CT.indptr)
        cap = max(int(self.max_df_frac * C.shape[0]), 50)
        drop = np.flatnonzero(df > cap)
        if len(drop):
            mask = np.ones(CT.shape[0], dtype=np.float32)
            mask[drop] = 0.0
            CT = (sp.diags(mask) @ CT).tocsr()
            CT.eliminate_zeros()
        return CT

    def _plan_chunks(self, Q: sp.csr_matrix, CT: sp.csr_matrix) -> list[tuple[int, int]]:
        """Group consecutive queries so each chunk's work (sum over query n-grams of
        posting-list length) stays under max_work_per_chunk."""
        df = np.diff(CT.indptr).astype(np.float64)
        Qb = Q.copy()
        Qb.data[:] = 1.0
        work = np.asarray(Qb @ df).ravel()
        self.last_total_work = float(work.sum())
        ranges, start, acc = [], 0, 0.0
        for i, w in enumerate(work):
            if i > start and (acc + w > self.max_work_per_chunk
                              or i - start >= self.max_rows_per_chunk):
                ranges.append((start, i))
                start, acc = i, 0.0
            acc += w
        if start < len(work):
            ranges.append((start, len(work)))
        return ranges

    def search(self, Q: sp.csr_matrix, CT: sp.csr_matrix, k: int | None = None):
        """Top-k pool rows per query row. Returns (idx[nq,k] with -1 padding, scores)."""
        k = k or self.k
        if Q.shape[0] == 0 or CT.shape[1] == 0:
            return (np.full((Q.shape[0], k), -1, np.int64),
                    np.zeros((Q.shape[0], k), np.float32))
        ranges = self._plan_chunks(Q, CT)
        parts = fork_map(_search_worker, ranges, self.n_jobs,
                         shared={"Q": Q, "CT": CT, "k": k})
        return (np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts]))
