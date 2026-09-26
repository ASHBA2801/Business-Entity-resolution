# Multi-Source Business Entity Resolution (ML Challenge 2026)

An end-to-end, high-performance machine learning pipeline for **Business Entity Resolution (ER)** across multiple noisy, heterogeneous data sources.

---

## 📌 Executive Summary & Problem Overview

Commercial enterprise databases aggregate business entity information from multiple disjoint sources (e.g., public registries, directories, partner feeds). These sources lack shared unique identifiers and feature pervasive data noise:
* **Name Variations:** Legal suffix inconsistencies (`Corp` vs. `Corporation`, `Pvt Ltd` vs. `Private Limited`), DBAs/trade names, native script transliterations, and typos.
* **Address Inconsistencies:** Landmark-based descriptions, missing PIN/postal codes or states, component reordering, and varying municipal formats.
* **Open-Set Partitions:** The training set covers `US` and `India`, whereas the evaluation test set introduces unseen geographic partitions (e.g., `France`).

### Task Objective
Given:
- **Source 1 ($S_1$):** Deduplicated reference dataset.
- **Source 2 ($S_2$) & Source 3 ($S_3$):** Noisy secondary sources.

For each record in $S_1$, identify all matching records in $S_2$ and $S_3$ (zero, one, or multiple matches), optimizing for the **$F_{0.5}$ metric** (which places double the emphasis on Precision over Recall).

---

## 🏗️ System Architecture (3-Stage Cascaded Pipeline)

```
                            ┌────────────────────────────────────────┐
                            │ Raw Datasets (S1, S2, S3 TSV Files)   │
                            └───────────────────┬────────────────────┘
                                                │
                                                ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: HYBRID BLOCKING & CANDIDATE GENERATION                                                  │
 │                                                                                                  │
 │   Rule-Based Normalization (Legal Suffixes, Address Abbreviations, Script Transliteration)       │
 │                                      │                                                           │
 │         ┌────────────────────────────┴────────────────────────────┐                              │
 │         ▼                                                         ▼                              │
 │  Sparse Inverted Index (TF-IDF)                            Dense Embedding Retrieval             │
 │  - Country-partitioned char n-grams (2,4)                  - Pretrained BGE-M3 (568M params)     │
 │  - Dual channels: Name+Address (top-20), Name (top-10)     - FAISS Index (top-20)                │
 │         └────────────────────────────┬────────────────────────────┘                              │
 │                                      ▼                                                           │
 │                Union, Deduplication & Detail Parquet Export                                      │
 │                Output: candidate_pairs.tsv (~30-50 candidates / S1)                              │
 └──────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                        │
                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2: LIGHTGBM PAIR FILTERING & RANKING                                                       │
 │                                                                                                  │
 │  - 50+ engineered signals: Lexical similarities (Levenshtein, Jaccard, Token Sort/Set),         │
 │    Phonetic matching, Dense Cosine, Cross-S1 competition features, and Source Priors.            │
 │  - Grouped Out-Of-Fold (OOF) cross-validation preventing target leakage.                         │
 │  - Reduces candidates to a hard, high-precision Top-K per S1.                                    │
 └──────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                        │
                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3: TRANSFORMER CROSS-ENCODER RERANKING & THRESHOLDING                                      │
 │                                                                                                  │
 │  - DeBERTa-v3-base Cross-Encoder (Full fine-tuning or LoRA parameter-efficient adaptation).      │
 │  - Joint cross-attention: "[CLS] name1 | addr1 | country [SEP] name2 | addr2 | country [SEP]"   │
 │  - Classification head fusing deep text representations with LightGBM contextual metadata.       │
 │  - Platt probability calibration & threshold sweep optimized for F_0.5.                          │
 │  - Output: matching_results.tsv                                                                  │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt            # Python environment dependencies
│       ├── README.md                   # Stage-1 blocking specific documentation
│       └── src/
│           ├── preprocessing.py        # Entity normalization & transliteration
│           ├── sparse_retrieval.py     # TF-IDF inverted index & n-gram blocking
│           ├── dense_retrieval.py      # BGE-M3 embedding generation & FAISS search
│           ├── blocking_pipeline.py    # Stage 1 orchestration & candidate union
│           ├── feature_engineering.py  # 50+ pairwise lexical, phonetic & dense features
│           ├── stage2_lgbm.py          # Stage 2 LightGBM candidate ranker & filter
│           ├── stage3_cross_encoder.py # Stage 3 DeBERTa-v3 cross-encoder
│           ├── evaluate.py             # Candidate recall, precision & F_0.5 scorer
│           ├── make_subsample.py       # Relationship-preserving prototyping tool
│           └── utils.py                # Fast TSV I/O, parallel multiprocessing utilities
├── Dataset/                            # Input dataset directory (ignored by git)
│   └── student_resource/
│       ├── dataset/
│       │   ├── train/                  # Training split (S1, S2, S3, ground truth)
│       │   ├── val/                    # Validation split (S1, ground truth)
│       │   └── test/                   # Test split (S1, S2, S3)
│       └── utils/
│           └── validate_submission.py  # Official submission format validator
├── cache/                              # Intermediate embeddings & indices (ignored)
├── output/                             # Final candidate and matching TSVs (ignored)
├── .gitignore                          # Git ignore rules
└── README.md                           # Master project documentation (this file)
```

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.10+ (tested on Python 3.14)
* Recommended: CUDA-compatible GPU for accelerated embedding inference and transformer training (CPU execution is fully supported via `faiss-cpu` and multi-core CPU mapping).

### Environment Setup

```bash
# 1. Clone repository and navigate to root directory
cd "ML challenge-Final Execution"

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install required dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 🚀 Step-by-Step Execution Workflow

All pipeline commands should be run from `code/business_entity_resolution/`:

```bash
cd code/business_entity_resolution
```

### 1. Rapid Prototyping (Optional)
Create a lightweight, relationship-preserving subsample to verify the pipeline logic in seconds:

```bash
# Create subsample in cache
python src/make_subsample.py \
    --data-dir ../../Dataset/student_resource/dataset \
    --out-dir ../../cache/prototype_data

# Run Stage 1 blocking on prototype
python src/blocking_pipeline.py \
    --data-dir ../../cache/prototype_data \
    --queries val \
    --output-dir ../../cache/proto_out \
    --cache-dir ../../cache/proto_cache
```

### 2. Stage 1: Hybrid Blocking & Candidate Generation
Runs normalization, sparse TF-IDF retrieval, and dense BGE-M3 FAISS retrieval:

```bash
# Validation split candidate generation
python src/blocking_pipeline.py --queries val

# Test set candidate generation (generates output/candidate_pairs.tsv)
python src/blocking_pipeline.py --queries test
```
*Key Flag:* `--no-dense` allows running purely sparse channels when GPU or embedding caches are omitted.

### 3. Stage 2: LightGBM Pair Filtering
Extracts pairwise features and trains an out-of-fold LightGBM model to prune low-probability candidate pairs:

```bash
# Generate train_split candidates for Stage-2 training
python src/blocking_pipeline.py --queries train_split

# Train LightGBM ranker with 5-fold OOF CV
python src/stage2_lgbm.py --oof-folds 5
```

### 4. Stage 3: DeBERTa Cross-Encoder Reranking
Fine-tunes a cross-encoder model to score and calibrate hard candidate pairs:

```bash
# (Optional) Estimate dataset sizes, sequence lengths, and throughput
python src/stage3_cross_encoder.py --estimate-only

# Fine-tune with LoRA (parameter-efficient)
python src/stage3_cross_encoder.py --mode lora

# Or fine-tune full backbone
python src/stage3_cross_encoder.py --mode full
```

### 5. Final Output Generation & Evaluation
Score validation performance and output final test predictions:

```bash
# Evaluate F_0.5 on held-out validation set
python src/evaluate.py \
    --candidates output/val_candidate_pairs.tsv \
    --truth ../../Dataset/student_resource/dataset/val/val_ground_truth.tsv

# Score test set
python src/stage3_cross_encoder.py --score-only --model-dir output/stage3_model
```

---

## 📊 Evaluation Metric & Submission Format

### $F_{0.5}$ Metric
The competition evaluates entity matching using the precision-skewed $F_{0.5}$ score:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{(0.5^2 \times \text{Precision}) + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Submission Artifacts

Final submission files are generated in the `output/` directory:

1. **`matching_results.tsv`** *(Leaderboard-scored)*:
   ```tsv
   source1_entity_id	matched_entity_ids
   S1-00001	S2-00047,S2-00193,S3-00812
   S1-00002	S3-00004
   S1-00003	
   ```
2. **`candidate_pairs.tsv`** *(Stage 1 Candidate pool)*:
   ```tsv
   source1_entity_id	candidate_entity_ids
   S1-00001	S2-00047,S2-00193,S3-00812,S2-00999,S3-00100
   ...
   ```

### Local Validation
Verify output format integrity using the provided validator:

```bash
python ../../Dataset/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../Dataset/student_resource/dataset/test \
    --check-ids
```

---

## 🛡️ Fair Play & Compliance
* **Zero Target Leakage:** Training, validation, and test partitions are cleanly isolated.
* **Deterministic Seeds:** All stochastic components use explicit random seeds.
* **Open-Set Country Support:** Normalization and blocking dynamically handle open-set country partitions (e.g., France).
