# Business Entity Resolution — Stage 1: hybrid blocking

Candidate generation for matching each Source-1 entity to Source-2/3 records.
Output: `output/candidate_pairs.tsv` (`source1_entity_id<TAB>candidate_entity_ids`).

## Layout

```
src/
  preprocessing.py     rule-based normalization (legal suffixes, address abbreviations,
                       state names incl. native scripts, transliteration, junk stripping)
  sparse_retrieval.py  char n-gram TF-IDF blocking (country partition + pruned inverted index)
  dense_retrieval.py   pretrained BGE-M3 embeddings + FAISS (Flat / IVF-Flat / IVF-PQ)
  blocking_pipeline.py orchestration: normalize -> sparse + dense -> union -> TSV + report
  evaluate.py          candidate recall (per source/country/channel), reduction ratio
  make_subsample.py    relationship-preserving prototype dataset
  utils.py             TSV I/O, stage timing/memory, fork-based parallel map
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

BGE-M3 (`BAAI/bge-m3`, MIT license, 568M params) is downloaded from the Hugging Face
Hub on first use. No other network access is made; no external lookups of any kind.

## Run

Paths default to `../../Dataset/student_resource/dataset`, `../../output`, `../../cache`
(relative to this folder's parent project); override with `--data-dir`, `--output-dir`,
`--cache-dir`.

```bash
# 1. prototype on a small subsample (minutes)
python src/make_subsample.py --data-dir <dataset> --out-dir <cache>/prototype_data
python src/blocking_pipeline.py --data-dir <cache>/prototype_data --queries val \
       --output-dir <cache>/proto_out --cache-dir <cache>/proto_cache

# 2. recall on the held-out validation split (val S1 vs full train S2/S3 pool)
python src/blocking_pipeline.py --queries val

# 3. test-set candidates (writes output/candidate_pairs.tsv)
python src/blocking_pipeline.py --queries test
```

`--no-dense` runs the sparse channels only. Dense embeddings are cached in
`<cache>/dense_<split>_<source>.npy` and encoding resumes where it stopped.

## Method summary

* **Partitioning:** exact normalized `country` label (open set; an unseen label such as
  France becomes its own partition). 0 of 7.6M train true pairs cross countries.
* **Sparse channels:** `char_wb` TF-IDF, n-grams (2,4), sublinear TF, IDF fitted per
  country on a sample of all source records (train + test, unlabeled). Each record keeps
  its top-48 n-grams; n-grams in >2% of a pool partition are dropped from the index.
  Two channels: name + address (top-20 per source) and name only (top-10 per source).
* **Dense channel:** pretrained BGE-M3 on normalized text with native scripts kept,
  top-20 per source via FAISS.
* **Union:** deduplicated by entity id; per-channel score/rank saved to
  `<cache>/<split>_candidates_detail.parquet` for use as Stage-2 features.

## Fair play

Everything is rule-based or uses only the provided files. TF-IDF IDF statistics are
fitted on unlabeled source records (train + test); no ground truth is used for fitting,
and the validation ground truth is only used for evaluation.
