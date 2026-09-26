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

## 🏗️ System Architecture (4-Stage Cascaded Pipeline)

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
 │ STAGE 3: TRANSFORMER CROSS-ENCODER RERANKING                                                     │
 │                                                                                                  │
 │  - DeBERTa-v3-base Cross-Encoder (Full fine-tuning or LoRA parameter-efficient adaptation).      │
 │  - Joint cross-attention: "[CLS] name1 | addr1 | country [SEP] name2 | addr2 | country [SEP]"   │
 │  - Classification head fusing deep text representations with LightGBM contextual metadata.       │
 │  - Platt probability calibration on a held-out dev split.                                        │
 │  - Output: calibrated pair scores                                                                │
 └──────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                        │
                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 4: MATCH DECISION                                                                          │
 │                                                                                                  │
 │  - Score column / threshold / per-S1 cap chosen on val for macro F_0.5.                          │
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
│       ├── README.md                   # Per-stage CLI input/output contracts
│       └── src/
│           ├── preprocessing.py        # Entity normalization & transliteration
│           ├── sparse_retrieval.py     # TF-IDF inverted index & n-gram blocking
│           ├── dense_retrieval.py      # BGE-M3 embedding generation & FAISS search
│           ├── stage1_blocking.py      # Stage 1 CLI: TF-IDF fit + hybrid blocking
│           ├── stage2_lgbm.py          # Stage 2 CLI: LightGBM candidate filter (train/predict)
│           ├── stage3_cross_encoder.py # Stage 3 CLI: DeBERTa-v3 cross-encoder (train/score)
│           ├── stage4_decision.py      # Stage 4 CLI: match decision rule (tune/apply)
│           ├── feature_engineering.py  # 63 pairwise lexical, address & dense features
│           ├── evaluate.py             # Stage-1 candidate recall & reduction ratio
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

## 🚀 Execution Workflow

Each stage is a standalone CLI under `code/business_entity_resolution/src/` that takes
every input and output file as an explicit argument, so any stage can be run on its own
(including on another machine) given only its input files:

| Stage | Command | Key inputs → outputs |
|---|---|---|
| 1 | `stage1_blocking.py fit-vectorizers` / `block` | source TSVs → `candidate_pairs.tsv`, detail parquet, embeddings |
| 2 | `stage2_lgbm.py train` / `predict` | Stage-1 detail + source TSVs → filtered candidates (+ model dir) |
| 3 | `stage3_cross_encoder.py train` / `score` | Stage-2 filtered candidates + source TSVs → pair scores (+ model dir) |
| 4 | `stage4_decision.py tune` / `apply` | Stage-3 scores + S1 TSV → `decision.json` / `matching_results.tsv` |

The complete end-to-end command sequence and the exact input/output file contract of
every stage are in
[code/business_entity_resolution/README.md](code/business_entity_resolution/README.md).
For a quick check, run the same sequence on a small subsample created with
`src/make_subsample.py`.

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
