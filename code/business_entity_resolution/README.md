# Business Entity Resolution — four-stage pipeline

Matches each Source-1 entity to its Source-2/3 records:

| Stage | Script | What it does |
|---|---|---|
| 1 | `src/stage1_blocking.py` | hybrid blocking (char TF-IDF + BGE-M3/FAISS) → candidate pairs |
| 2 | `src/stage2_lgbm.py` | LightGBM pair scorer → small, hard top-K per S1 |
| 3 | `src/stage3_cross_encoder.py` | DeBERTa-v3 cross-encoder → calibrated pair scores |
| 4 | `src/stage4_decision.py` | decision rule tuned for macro F0.5 → `matching_results.tsv` |

Each stage is a standalone CLI. Every file it reads or writes is passed explicitly on
the command line; no stage relies on directory conventions or another stage's in-memory
state. Given only the input files listed in its contract below, any stage (or subcommand)
can run on a fresh machine. Each stage checks its declared inputs before doing any work.

## Layout

```
src/
  stage1_blocking.py     Stage 1 CLI: fit-vectorizers | block
  stage2_lgbm.py         Stage 2 CLI: train | predict
  stage3_cross_encoder.py Stage 3 CLI: train | score
  stage4_decision.py     Stage 4 CLI: tune | apply
  preprocessing.py       rule-based normalization (legal suffixes, address abbreviations,
                         state names incl. native scripts, transliteration, junk stripping)
  sparse_retrieval.py    char n-gram TF-IDF blocking (country partition + pruned inverted index)
  dense_retrieval.py     pretrained BGE-M3 embeddings + FAISS (Flat / IVF-Flat / IVF-PQ)
  feature_engineering.py Stage-2 pairwise features
  evaluate.py            Stage-1 candidate recall, reduction ratio
  make_subsample.py      relationship-preserving prototype dataset
  utils.py               TSV I/O, ground-truth loading, stage timing, fork-based parallel map
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Pretrained models come from the Hugging Face Hub on first use and are cached locally
(`~/.cache/huggingface`). On a fresh machine, the steps that need them either need
network access or a copied cache:

* `BAAI/bge-m3` (MIT, 568M params): Stage 1 `block`, but only when an embedding file
  is missing or incomplete.
* `microsoft/deberta-v3-base` (MIT): Stage 3 `train` and `score`. The base weights are
  needed even to score with a saved LoRA adapter.

No other network access is made; no external lookups of any kind.

## End-to-end run

All paths are explicit. The variables below are only for readability. Run from this
folder.

```bash
D=../../Dataset/student_resource/dataset   # challenge data
W=../../work                               # any working directory for artifacts
O=../../output                             # submission files

# ---- Stage 1 ------------------------------------------------------------------------
python src/stage1_blocking.py fit-vectorizers \
    --inputs $D/train/train_source{1,2,3}.tsv $D/test/test_source{1,2,3}.tsv \
    --output $W/s1/vectorizers.pkl

# one `block` per S1 file; train_split and val share the train pool (and its embeddings)
for SPLIT in train_split val; do
  python src/stage1_blocking.py block \
      --s1 $D/$SPLIT/${SPLIT}_source1.tsv --truth $D/$SPLIT/${SPLIT}_ground_truth.tsv \
      --s2 $D/train/train_source2.tsv --s3 $D/train/train_source3.tsv \
      --vectorizers $W/s1/vectorizers.pkl \
      --s1-emb $W/emb/${SPLIT}_S1.npy --s2-emb $W/emb/train_S2.npy --s3-emb $W/emb/train_S3.npy \
      --output $W/s1/${SPLIT}_candidate_pairs.tsv \
      --detail-output $W/s1/${SPLIT}_detail.parquet --report $W/s1/${SPLIT}_report.json
done
python src/stage1_blocking.py block \
    --s1 $D/test/test_source1.tsv --s2 $D/test/test_source2.tsv --s3 $D/test/test_source3.tsv \
    --vectorizers $W/s1/vectorizers.pkl \
    --s1-emb $W/emb/test_S1.npy --s2-emb $W/emb/test_S2.npy --s3-emb $W/emb/test_S3.npy \
    --output $O/candidate_pairs.tsv --detail-output $W/s1/test_detail.parquet

# ---- Stage 2 ------------------------------------------------------------------------
TRAIN_POOL="--s2 $D/train/train_source2.tsv --s3 $D/train/train_source3.tsv \
            --s2-emb $W/emb/train_S2.npy --s3-emb $W/emb/train_S3.npy"
TEST_POOL="--s2 $D/test/test_source2.tsv --s3 $D/test/test_source3.tsv \
           --s2-emb $W/emb/test_S2.npy --s3-emb $W/emb/test_S3.npy"

python src/stage2_lgbm.py train $TRAIN_POOL --vectorizers $W/s1/vectorizers.pkl \
    --train-s1 $D/train_split/train_split_source1.tsv \
    --train-truth $D/train_split/train_split_ground_truth.tsv \
    --train-detail $W/s1/train_split_detail.parquet --train-s1-emb $W/emb/train_split_S1.npy \
    --eval-s1 $D/val/val_source1.tsv --eval-truth $D/val/val_ground_truth.tsv \
    --eval-detail $W/s1/val_detail.parquet --eval-s1-emb $W/emb/val_S1.npy \
    --model-dir $W/s2/model \
    --eval-output $W/s2/val_filtered.tsv --oof-output $W/s2/train_split_filtered.tsv

python src/stage2_lgbm.py predict $TEST_POOL --vectorizers $W/s1/vectorizers.pkl \
    --model-dir $W/s2/model --s1 $D/test/test_source1.tsv \
    --detail $W/s1/test_detail.parquet --s1-emb $W/emb/test_S1.npy \
    --output $W/s2/test_filtered.tsv

# ---- Stage 3 ------------------------------------------------------------------------
python src/stage3_cross_encoder.py train \
    --s2 $D/train/train_source2.tsv --s3 $D/train/train_source3.tsv \
    --train-candidates $W/s2/train_split_filtered.tsv \
    --train-s1 $D/train_split/train_split_source1.tsv \
    --train-truth $D/train_split/train_split_ground_truth.tsv \
    --val-candidates $W/s2/val_filtered.tsv \
    --val-s1 $D/val/val_source1.tsv --val-truth $D/val/val_ground_truth.tsv \
    --model-dir $W/s3/lora --val-output $W/s3/val_scores.tsv --mode lora
    # add --estimate-only (+ --full-{train,val,test}-s1 <full S1 TSVs>) for a cost check

python src/stage3_cross_encoder.py score \
    --s2 $D/test/test_source2.tsv --s3 $D/test/test_source3.tsv \
    --model-dir $W/s3/lora --candidates $W/s2/test_filtered.tsv \
    --s1 $D/test/test_source1.tsv --output $W/s3/test_scores.tsv

# ---- Stage 4 ------------------------------------------------------------------------
python src/stage4_decision.py tune \
    --scores $W/s3/val_scores.tsv --s1 $D/val/val_source1.tsv \
    --truth $D/val/val_ground_truth.tsv --output $W/s4/decision.json

python src/stage4_decision.py apply \
    --scores $W/s3/test_scores.tsv --s1 $D/test/test_source1.tsv \
    --decision $W/s4/decision.json --output $O/matching_results.tsv
```

**Prototype:** `python src/make_subsample.py --data-dir $D --out-dir <proto>` writes
the same directory layout at a few thousand rows. Set `D=<proto>` and the whole recipe
above runs in minutes.

## Stage contracts

"Req." marks required arguments. Every output's parent directory is created if it
doesn't exist. File formats are defined in the next section. Two rules hold across
stages:

* **Consistent pool files.** Wherever a stage takes `--s1/--s2/--s3` (or
  `--train-s1`, etc.) alongside an upstream artifact, pass the same source files the
  artifact was built from. For example, Stage 2's `--s2/--s3` must be the pool Stage 1
  matched that S1 against. Artifacts refer to records by `entity_id`, and a stage fails
  if an id is missing from the given files.
* **One vectorizer pickle.** Stage 2 `train` and `predict` must receive the same
  vectorizer pickle, because the TF-IDF features depend on it.

### Stage 1 — `stage1_blocking.py`

**`fit-vectorizers`**: fits one char TF-IDF per (text field, country), unsupervised.

| | Argument | File | Req. |
|---|---|---|---|
| in | `--inputs F [F ...]` | source TSVs. Pass all six train+test S1/S2/S3 files, so every country that `block` will meet gets a vectorizer | ✓ |
| out | `--output` | vectorizers pickle | ✓ |

Options: `--fit-sample-per-file` (150 000 rows sampled per file, seed 0),
`--ngram-range 2 4`, `--min-df 2`, `--n-jobs`.

**`block`**: candidates for one S1 file against one S2/S3 pool.

| | Argument | File | Req. |
|---|---|---|---|
| in | `--s1` | source TSV (queries) | ✓ |
| in | `--s2`, `--s3` | source TSVs (pool) | ✓ |
| in | `--vectorizers` | vectorizers pickle; must cover every country in `--s1` | ✓ unless `--no-sparse` |
| in | `--truth` | ground-truth TSV for `--s1`; adds recall to the report | |
| in/out | `--s1-emb`, `--s2-emb`, `--s3-emb` | embedding files. A complete file whose fingerprint matches the texts is reused; otherwise it is (re)encoded and written, resuming where a previous run stopped. If omitted, embeddings are computed in memory and not saved, so Stage 2's `bge_m3_cos` will be NaN | dense only |
| out | `--output` | candidate pairs TSV | ✓ |
| out | `--detail-output` | candidates detail parquet (Stage-2 input) | for Stage 2 |
| out | `--report` | JSON: counts, recall by source/country/channel, runtime, memory | |

Options: `--no-dense`, `--no-sparse`, `--k-sparse 20`, `--k-sparse-name 10`,
`--k-dense 20`, `--top-features 48`, `--max-df-frac 0.02`, `--dense-model`,
`--dense-batch-size`, `--dense-max-len`, `--nprobe`, `--n-jobs`.

### Stage 2 — `stage2_lgbm.py`

Both subcommands take the pool and TF-IDF inputs:

| | Argument | File | Req. |
|---|---|---|---|
| in | `--s2`, `--s3` | source TSVs of the pool Stage 1 used | ✓ |
| in | `--s2-emb`, `--s3-emb` | Stage-1 embedding files of `--s2`/`--s3` | |
| in | `--vectorizers` | Stage-1 vectorizers pickle | ✓ |

An embedding file is only used if it is complete and has exactly as many rows as its
source TSV. Otherwise `bge_m3_cos` is NaN for those records, and `predict` warns if the
model was trained with embeddings.

**`train`**: fits the model on train_split pairs, early-stops on a grouped holdout of
train_split, and evaluates on val.

| | Argument | File | Req. |
|---|---|---|---|
| in | `--train-s1`, `--train-truth`, `--train-detail` | train_split source TSV, its ground truth, its Stage-1 detail parquet | ✓ |
| in | `--train-s1-emb` | Stage-1 embedding file of `--train-s1` | |
| in | `--eval-s1`, `--eval-truth`, `--eval-detail` | the same, for val | ✓ |
| in | `--eval-s1-emb` | Stage-1 embedding file of `--eval-s1` | |
| out | `--model-dir` | Stage-2 model dir | ✓ |
| out | `--eval-output` | filtered candidates TSV for val (Stage-3 `--val-candidates`) | ✓ |
| out | `--oof-output` | out-of-fold filtered candidates TSV for train_split (Stage-3 `--train-candidates`) | for Stage 3 |

Options: `--oof-folds 5`, `--top-k 3`, `--score-floor 0.02`, `--no-match-threshold 0.05`
(all three filter settings are stored in `meta.json`), `--lr`, `--num-leaves`,
`--max-train-s1`, `--es-frac`, `--no-ablation`, `--seed`, `--n-jobs`.

**`predict`**: scores and filters one S1 file's Stage-1 candidates.

| | Argument | File | Req. |
|---|---|---|---|
| in | `--model-dir` | Stage-2 model dir (`lgbm.txt` + `meta.json`) | ✓ |
| in | `--s1`, `--detail` | source TSV and its Stage-1 detail parquet | ✓ |
| in | `--s1-emb` | Stage-1 embedding file of `--s1` | |
| in | `--truth` | ground truth for `--s1`; adds filter statistics to the report | |
| out | `--output` | filtered candidates TSV | ✓ |
| out | `--report` | JSON: pair counts, likely-no-match count, mean score by country | |

`--top-k`, `--score-floor` and `--no-match-threshold` default to the values stored in
`meta.json`.

### Stage 3 — `stage3_cross_encoder.py`

Both subcommands take `--s2`, `--s3` (✓, the pool the candidates came from), plus
`--device`, `--eval-batch-size`, `--[no-]bf16` and `--n-jobs`.

**`train`**: fine-tunes on the out-of-fold train_split pairs. Early stopping and Platt
calibration use a grouped dev split of train_split, and val is scored for reporting.

| | Argument | File | Req. |
|---|---|---|---|
| in | `--train-candidates` | Stage-2 **out-of-fold** filtered candidates TSV for train_split | ✓ |
| in | `--train-s1`, `--train-truth` | train_split source TSV, its ground truth | ✓ |
| in | `--val-candidates` | Stage-2 filtered candidates TSV for val | ✓ |
| in | `--val-s1`, `--val-truth` | val source TSV, its ground truth | ✓ |
| in | `--full-train-s1`, `--full-val-s1`, `--full-test-s1` | full-size S1 TSVs, used only for their row counts in the cost projection | |
| out | `--model-dir` | Stage-3 model dir | ✓ |
| out | `--val-output` | scores TSV for val (Stage-4 `tune` input) | ✓ unless `--estimate-only` |

Options: `--mode lora|full`, `--estimate-only` (token lengths, throughput and GPU-hour
cost, then exits), `--model-name`, `--epochs`, `--max-steps`, `--batch-size`, `--lr`,
`--max-len`, `--[no-]use-feats`, `--max-train-s1`, `--dev-frac`, `--patience`, `--seed`.

**`score`**: scores one filtered candidates file with a trained model.

| | Argument | File | Req. |
|---|---|---|---|
| in | `--model-dir` | Stage-3 model dir (must contain calibration, i.e. a finished `train`) | ✓ |
| in | `--candidates` | Stage-2 filtered candidates TSV | ✓ |
| in | `--s1` | source TSV of `--candidates` | ✓ |
| in | `--truth` | ground truth for `--s1`; adds the threshold sweep to the report | |
| out | `--output` | scores TSV | ✓ |
| out | `--report` | JSON | |

### Stage 4 — `stage4_decision.py`

Decision rule, per S1: rank its scored candidates by `score_column` and keep those with
score ≥ `threshold`, at most `max_matches` of them (0 = no cap). An S1 with nothing kept,
or with no scored pairs at all, is predicted as a singleton (empty list).

**`tune`**: grid-searches `score_column` (`raw_score`, `normalized_score`) ×
`threshold` (0.05–0.995) × `max_matches` (1, 2, 3, no cap). It keeps the rule with the
best macro F0.5 over every S1 in `--s1`; ties go to the higher threshold, then the
tighter cap. The val score this reports is optimistic, because the rule was chosen on
val.

| | Argument | File | Req. |
|---|---|---|---|
| in | `--scores` | scores TSV (val) | ✓ |
| in | `--s1` | source TSV of the scored split | ✓ |
| in | `--truth` | ground truth for `--s1` | ✓ |
| out | `--output` | decision JSON | ✓ |
| out | `--matches-output` | matching TSV for `--s1` under the chosen rule | |
| out | `--sweep-output` | TSV of every rule tried and its macro F0.5 | |

**`apply`**

| | Argument | File | Req. |
|---|---|---|---|
| in | `--scores` | scores TSV | ✓ |
| in | `--s1` | source TSV; defines the output rows (every S1 appears once) | ✓ |
| in | `--decision` | decision JSON, or a manual rule: `--threshold` [`--score-column`, `--max-matches`] | ✓ (either) |
| in | `--truth` | ground truth for `--s1`; prints macro F0.5 | |
| out | `--output` | matching TSV (`matching_results.tsv`) | ✓ |

## File formats

All TSVs are UTF-8, tab-separated, with a header row. Id lists are comma-joined with no
spaces, and an empty cell means an empty list.

| File | Written by | Content |
|---|---|---|
| source TSV | challenge data | `entity_id, business_name, business_address, country` |
| ground-truth TSV | challenge data | `source1_entity_id, matched_entity_ids` |
| vectorizers pickle | S1 `fit-vectorizers` | `dict`: `(field, country) → sklearn TfidfVectorizer` for field ∈ {`sparse_text`, `name_text`}, and `"_meta"` → fit settings |
| embedding file | S1 `block` | `X.npy`: float16 `[rows, 1024]`, L2-normalized, row *i* = row *i* of its source TSV; `X.npy.json`: `{fingerprint, n, dim, done}` (complete when `done == n`) |
| candidate pairs TSV | S1 `block` | `source1_entity_id, candidate_entity_ids`; one row per S1 |
| candidates detail parquet | S1 `block` | one row per (S1, candidate): `source1_entity_id, candidate_entity_id`, and `<ch>_score, <ch>_rank` for ch ∈ {`sparse`, `sparse_name`, `dense`}. Rank 0 / score NaN = not retrieved by that channel |
| Stage-2 model dir | S2 `train` | `lgbm.txt` (LightGBM), `meta.json` (feature list, filter settings, which embeddings were used), `stage2_report.json`, `stage2_feature_importance.tsv`, `stage2_pr_curve.png` |
| filtered candidates TSV | S2 `train` / `predict` | `source1_entity_id, candidate_entity_ids, candidate_scores, best_score, likely_no_match`; one row per S1; candidates best-first, with scores in matching order; `likely_no_match` = 1 when no candidate passed |
| Stage-3 model dir | S3 `train` | `adapter/` (LoRA) or `backbone/` (full), `head.pt`, `meta.json` (base model, max length, feature standardization, Platt `a`/`b`), `stage3_report.json`, `stage3_val_sweep.tsv`, `stage3_cost_estimate.json` |
| scores TSV | S3 `train` / `score` | one row per pair: `source1_entity_id, candidate_entity_id, raw_score` (calibrated P(match)), `normalized_score` (share against the S1's other candidates plus a no-match slot), `raw_logit, no_match_score, lgbm_score, lgbm_rank` |
| decision JSON | S4 `tune` | `{score_column, threshold, max_matches, tuned_macro_f05, tuned_on, all_empty_macro_f05}` |
| matching TSV | S4 `apply` | `source1_entity_id, matched_entity_ids`; one row per S1 in `--s1` order |

## Method summary

* **Partitioning:** exact normalized `country` label (open set; an unseen label such as
  France becomes its own partition). 0 of 7.6M train true pairs cross countries.
* **Sparse channels:** `char_wb` TF-IDF, n-grams (2,4), sublinear TF, IDF fitted per
  country on a sample of all source records (train + test, unlabeled). Each record keeps
  its top-48 n-grams; n-grams in >2% of a pool partition are dropped from the index.
  Two channels: name + address (top-20 per source) and name only (top-10 per source).
* **Dense channel:** pretrained BGE-M3 on normalized text with native scripts kept,
  top-20 per source via FAISS.
* **Union:** deduplicated by entity id; per-channel score/rank saved to the detail
  parquet for use as Stage-2 features.
* **Stage 2:** 63 pairwise features (string, address, TF-IDF, BGE-M3 cosine, competition
  context); keeps top-3 with score ≥ 0.02 and flags an S1 as likely-no-match when its best
  score is < 0.05. Out-of-fold scores for train_split give Stage 3 the same kind of pairs
  it sees at test time.
* **Stage 3:** DeBERTa-v3-base cross-encoder (LoRA or full), with Stage-2 context
  features fused at the head and Platt-calibrated on a train_split dev split.
* **Stage 4:** a single score/threshold/cap rule chosen on val for the challenge's macro
  F0.5.

## Fair play

Everything is rule-based or uses only the provided files. TF-IDF IDF statistics are
fitted on unlabeled source records (train + test); no ground truth is used for fitting.
The val ground truth is used for evaluation and, in Stage 4, only to choose the
decision rule.
