"""Dense blocking with a pretrained BGE-M3 encoder (no fine-tuning) + FAISS ANN search.

Embeddings are cached on disk as fp16 memmaps and written block by block, so a long
encoding run (millions of records on CPU) can be interrupted and resumed.

Index choice per pool (inner product on L2-normalized vectors == cosine):
  * n <= flat_max                      -> IndexFlatIP (exact; cheap at prototype scale)
  * n * d * 4 bytes <= ivf_flat_budget -> IVF-Flat (exact vectors, probe nprobe lists)
  * otherwise                          -> IVF-PQ (compressed; the only option that fits
    multi-million pools in RAM on a 14 GB machine; HNSW would need the full fp32
    vectors plus graph links, i.e. >12 GB for a 3M pool, and builds slowly on CPU)
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np


class DenseBlocker:
    def __init__(self, model_name="BAAI/bge-m3", batch_size=64, max_seq_length=64,
                 device=None, k=20, flat_max=200_000, ivf_flat_budget=3e9,
                 nprobe=32, pq_m=64, n_threads=None, encode_block=8_192):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device
        self.k = k
        self.flat_max = flat_max
        self.ivf_flat_budget = ivf_flat_budget
        self.nprobe = nprobe
        self.pq_m = pq_m
        self.n_threads = n_threads
        self.encode_block = encode_block
        self._model = None

    # -- encoding ----------------------------------------------------------------
    def _load_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.n_threads and self.device == "cpu":
                torch.set_num_threads(self.n_threads)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = self.max_seq_length
            if self.device.startswith("cuda"):
                self._model.half()  # fp16 on GPU
        return self._model

    def _fingerprint(self, texts: list[str]) -> str:
        h = hashlib.md5()
        h.update(f"{self.model_name}|{self.max_seq_length}|{len(texts)}".encode())
        for t in texts:
            h.update(t.encode("utf-8", "replace"))
            h.update(b"\x00")
        return h.hexdigest()

    def encode(self, texts: list[str], cache_path: str | None = None) -> np.ndarray:
        """L2-normalized embeddings, float32 [n, d]. Cached/resumable if cache_path."""
        n = len(texts)
        if n == 0:
            return np.zeros((0, 1024), dtype=np.float32)
        if cache_path is None:
            m = self._load_model()
            return m.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

        meta_path = cache_path + ".json"
        fp = self._fingerprint(texts)
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("fingerprint") != fp:
                meta = None  # texts/model changed -> re-encode
        if meta is None:
            m = self._load_model()
            dim = (m.get_embedding_dimension() if hasattr(m, "get_embedding_dimension")
                   else m.get_sentence_embedding_dimension())
            meta = {"fingerprint": fp, "n": n, "dim": dim, "done": 0}
            np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16, shape=(n, dim))
            self._write_meta(meta_path, meta)

        if meta["done"] < n:
            m = self._load_model()
            out = np.load(cache_path, mmap_mode="r+")
            for s in range(meta["done"], n, self.encode_block):
                e = min(s + self.encode_block, n)
                out[s:e] = m.encode(texts[s:e], batch_size=self.batch_size,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False).astype(np.float16)
                out.flush()
                meta["done"] = e
                self._write_meta(meta_path, meta)
                print(f"    encoded {e}/{n}", flush=True)
            del out
        emb = np.load(cache_path, mmap_mode="r")
        return emb

    @staticmethod
    def _write_meta(path, meta):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, path)

    # -- indexing / search ---------------------------------------------------------
    def build_index(self, emb: np.ndarray):
        import faiss
        if self.n_threads:
            faiss.omp_set_num_threads(self.n_threads)
        n, d = emb.shape
        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731 (per block)
        if n <= self.flat_max:
            index = faiss.IndexFlatIP(d)
            index.add(f32(emb))
            self.last_index_type = "flat"
            return index
        nlist = int(min(max(4 * np.sqrt(n), 256), n // 39))
        quantizer = faiss.IndexFlatIP(d)
        if n * d * 4 <= self.ivf_flat_budget:
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.last_index_type = f"ivfflat(nlist={nlist})"
        else:
            index = faiss.IndexIVFPQ(quantizer, d, nlist, self.pq_m, 8,
                                     faiss.METRIC_INNER_PRODUCT)
            self.last_index_type = f"ivfpq(nlist={nlist},m={self.pq_m})"
        rng = np.random.default_rng(0)
        train_rows = np.sort(rng.choice(n, size=min(n, 64 * nlist), replace=False))
        index.train(f32(emb[train_rows]))
        for s in range(0, n, 200_000):  # add in blocks to bound temp memory
            index.add(f32(emb[s:s + 200_000]))
        index.nprobe = self.nprobe
        return index

    def search(self, index, q: np.ndarray, k: int | None = None):
        k = k or self.k
        nq = q.shape[0]
        if nq == 0 or index.ntotal == 0:
            return np.full((nq, k), -1, np.int64), np.zeros((nq, k), np.float32)
        scores, idx = index.search(np.ascontiguousarray(q, dtype=np.float32), min(k, index.ntotal))
        if idx.shape[1] < k:  # pool smaller than k
            pad = k - idx.shape[1]
            idx = np.hstack([idx, np.full((nq, pad), -1, np.int64)])
            scores = np.hstack([scores, np.zeros((nq, pad), np.float32)])
        return idx.astype(np.int64), scores.astype(np.float32)
