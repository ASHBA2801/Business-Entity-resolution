"""Stage 3: DeBERTa-v3-base cross-encoder over Stage-2's filtered top-K pairs.

Two subcommands; every input and output is an explicit path (see README for the contract):

  train   Stage-2 out-of-fold filtered train_split candidates + Stage-2 filtered val
          candidates + source TSVs + ground truth -> model dir (weights, calibration,
          reports) + val pair scores (Stage-4 tuning input)
  score   model dir + any Stage-2 filtered candidates TSV (e.g. test) -> pair scores

    python src/stage3_cross_encoder.py train ... --estimate-only   # rows, lengths, cost
    python src/stage3_cross_encoder.py train ... --mode lora
    python src/stage3_cross_encoder.py score --model-dir <dir> ...

Data design
  * Training pairs = Stage 2's top-K for train_split S1, scored OUT-OF-FOLD, i.e. the
    same kind of hard pairs Stage 2 passes at test time. Labels from train_split truth.
  * Grouped split of the training S1s into fit / dev. Early stopping (pair-level F0.5 at
    the best threshold) and Platt calibration use dev.
  * val (disjoint S1s, filtered by the deployed Stage-2 model) is never used for training,
    early stopping or calibration: threshold sweeps / no-match analysis are reported there.

Input: "[CLS] name1 | address1 | country [SEP] name2 | address2 | country [SEP]" built from
preprocessing.light_name / light_address (noise cleaned, abbreviations and legal forms kept
as written), optionally fused with Stage-2 context features at the classification head.

Model dir (written by train, read by score):
  adapter/ (LoRA) or backbone/ (full) + head.pt + meta.json (calibration, max_len, feature
  standardization), stage3_report.json, stage3_val_sweep.tsv, stage3_cost_estimate.json
Scores TSV: source1_entity_id, candidate_entity_id, raw_score, normalized_score, raw_logit,
  no_match_score, lgbm_score, lgbm_rank
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import light_address, light_name, normalize_country  # noqa: E402
from utils import (chunk_ranges, default_n_jobs, ensure_parent, fork_map, get_shared,  # noqa: E402
                   load_truth, read_tsv, require_files, stage)

FEATS = ["lgbm_score", "lgbm_logit", "lgbm_rank", "lgbm_gap", "n_cands", "cand_is_s3"]
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95,
              0.97, 0.98, 0.99]
LORA_TARGETS = ["query_proj", "key_proj", "value_proj", "dense"]


# =========================================================================== data
def read_filtered(path: str) -> tuple[list[str], pd.DataFrame]:
    """Stage-2 filtered TSV -> (every S1 id in the file, one row per passed pair) with the
    Stage-2 context features of each pair."""
    df = read_tsv(path)
    s1, cand, sc, rk = [], [], [], []
    for s, ids, scores in zip(df["source1_entity_id"], df["candidate_entity_ids"],
                              df["candidate_scores"]):
        if not ids:
            continue
        for r, (c, x) in enumerate(zip(ids.split(","), scores.split(",")), 1):
            s1.append(s); cand.append(c); sc.append(float(x)); rk.append(r)
    pairs = pd.DataFrame({"s1": s1, "cand": cand, "lgbm_score": np.array(sc, np.float32),
                          "lgbm_rank": np.array(rk, np.float32)})
    g = pairs.groupby("s1", sort=False)["lgbm_score"]
    pairs["lgbm_gap"] = (g.transform("max") - pairs["lgbm_score"]).astype(np.float32)
    pairs["n_cands"] = g.transform("size").astype(np.float32)
    p = pairs["lgbm_score"].clip(1e-4, 1 - 1e-4)
    pairs["lgbm_logit"] = np.log(p / (1 - p)).astype(np.float32)
    pairs["cand_is_s3"] = pairs["cand"].str.startswith("S3-").astype(np.float32)
    return df["source1_entity_id"].tolist(), pairs


def _text_worker(rng):
    s, e = rng
    n, a, c = get_shared("n")[s:e], get_shared("a")[s:e], get_shared("c")[s:e]
    return [f"{light_name(x)} | {light_address(y)} | {normalize_country(z)}"
            for x, y, z in zip(n, a, c)]


def load_texts(paths: list[str], ids: set, n_jobs: int) -> dict[str, str]:
    """Cross-encoder text for every record in `ids`, read from the given source files."""
    out = {}
    for p in paths:
        df = read_tsv(p)
        df = df[df["entity_id"].isin(ids)]
        cols = {k: df[c].tolist() for k, c in
                (("n", "business_name"), ("a", "business_address"), ("c", "country"))}
        parts = fork_map(_text_worker, chunk_ranges(len(df), 20_000), n_jobs, shared=cols)
        out.update(zip(df["entity_id"], (t for part in parts for t in part)))
    return out


def build_split(name: str, candidates: str, s1: str, s2: str, s3: str, truth: str | None,
                n_jobs: int) -> dict:
    """Pairs + texts (+ labels if truth is given) for one Stage-2 filtered candidates file."""
    require_files(candidates, s1, s2, s3, truth)
    all_s1, pairs = read_filtered(candidates)
    truth = load_truth(truth) if truth else None
    ids = set(pairs["s1"]) | set(pairs["cand"])
    texts = load_texts([s1, s2, s3], ids, n_jobs)
    missing = ids - texts.keys()
    if missing:
        raise ValueError(f"{name}: {len(missing)} ids in {candidates} not in the given source "
                         f"files, e.g. {sorted(missing)[:3]}")
    pairs["text_a"] = pairs["s1"].map(texts)
    pairs["text_b"] = pairs["cand"].map(texts)
    if truth is not None:
        pairs["label"] = np.fromiter((c in truth.get(s, ()) for s, c in
                                      zip(pairs["s1"], pairs["cand"])), np.float32, len(pairs))
    return {"name": name, "all_s1": all_s1, "pairs": pairs, "truth": truth}


def grouped_fit_dev(pairs: pd.DataFrame, cfg) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    s1 = pairs["s1"].unique()
    if cfg.max_train_s1 and len(s1) > cfg.max_train_s1:
        s1 = rng.choice(s1, cfg.max_train_s1, replace=False)
    n_dev = min(int(len(s1) * cfg.dev_frac), cfg.max_dev_s1) if cfg.max_dev_s1 else int(len(s1) * cfg.dev_frac)
    dev = set(rng.choice(s1, n_dev, replace=False))
    m_dev = pairs["s1"].isin(dev).to_numpy()
    m_fit = pairs["s1"].isin(set(s1) - dev).to_numpy()
    assert not (set(pairs["s1"][m_fit]) & set(pairs["s1"][m_dev]))
    return m_fit, m_dev


# =========================================================================== tokens
def tokenize(tok, pairs: pd.DataFrame, max_len: int | None) -> list[np.ndarray]:
    enc = tok(pairs["text_a"].tolist(), pairs["text_b"].tolist(),
              truncation="longest_first" if max_len else False, max_length=max_len,
              return_attention_mask=False, return_token_type_ids=False)
    return [np.asarray(x, dtype=np.int32) for x in enc["input_ids"]]


def length_stats(tok, pairs: pd.DataFrame, n: int, seed: int) -> dict:
    """Token-length distribution of the untruncated pair input (and each side)."""
    smp = pairs.sample(n=min(n, len(pairs)), random_state=seed)
    both = np.array([len(x) for x in tokenize(tok, smp, None)])
    side = np.array([len(x) for x in tok(smp["text_a"].tolist() + smp["text_b"].tolist(),
                                         add_special_tokens=False)["input_ids"]])
    q = lambda v: {f"p{p}": int(np.percentile(v, p)) for p in (50, 90, 95, 99, 99.5, 99.9)}  # noqa: E731
    return {"n_sampled": int(len(smp)), "pair": {**q(both), "max": int(both.max()),
                                                  "mean": round(float(both.mean()), 1)},
            "one_side": {**q(side), "max": int(side.max())}, "_pair_lengths": both}


def choose_max_len(stats: dict, cfg) -> int:
    if cfg.max_len:
        return cfg.max_len
    L = int(math.ceil(stats["pair"]["p99.5"] / 8) * 8)
    return int(min(max(L, 32), 512))


# =========================================================================== model
def build_model(cfg, n_feats: int):
    import torch
    from torch import nn
    from transformers import AutoModel

    # fp32 master weights (the hub checkpoint loads as fp16 by default); bf16 autocast on GPU
    backbone = AutoModel.from_pretrained(cfg.model_name, dtype=torch.float32)
    if cfg.mode == "lora":
        from peft import LoraConfig, get_peft_model
        backbone = get_peft_model(backbone, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=LORA_TARGETS, bias="none"))
    elif cfg.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()

    class CrossEncoder(nn.Module):
        """[CLS] -> dense+GELU pooler -> concat Stage-2 features -> MLP -> logit."""

        def __init__(self):
            super().__init__()
            h = backbone.config.hidden_size
            self.backbone = backbone
            self.pooler = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, h), nn.GELU())
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h + n_feats, 256), nn.GELU(),
                                      nn.Linear(256, 1))

        def forward(self, input_ids, attention_mask, feats=None):
            cls = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state[:, 0]
            z = self.pooler(cls)
            if feats is not None and feats.shape[1]:
                z = torch.cat([z, feats.to(z.dtype)], dim=1)
            return self.head(z).squeeze(-1)

    return CrossEncoder()


def head_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith(("pooler.", "head."))}


def trainable_state(model) -> dict:
    """What changes during training: LoRA + head (small) or everything (full)."""
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def count_params(model) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_M": round(tot / 1e6, 2), "trainable_M": round(tr / 1e6, 3),
            "trainable_pct": round(100 * tr / tot, 3)}


# =========================================================================== batching
class Batches:
    """Dynamic padding; length-bucketed shuffling for training, length-sorted for inference."""

    def __init__(self, ids: list[np.ndarray], feats: np.ndarray, labels: np.ndarray | None,
                 pad_id: int, device):
        self.ids, self.feats, self.labels = ids, feats, labels
        self.lens = np.array([len(x) for x in ids])
        self.pad_id, self.device = pad_id, device

    def collate(self, idx: np.ndarray):
        import torch
        L = int(self.lens[idx].max())
        x = np.full((len(idx), L), self.pad_id, dtype=np.int64)
        m = np.zeros((len(idx), L), dtype=np.int64)
        for r, i in enumerate(idx):
            x[r, :self.lens[i]] = self.ids[i]
            m[r, :self.lens[i]] = 1
        b = {"input_ids": torch.from_numpy(x).to(self.device),
             "attention_mask": torch.from_numpy(m).to(self.device),
             "feats": torch.from_numpy(self.feats[idx]).to(self.device)}
        y = torch.from_numpy(self.labels[idx]).to(self.device) if self.labels is not None else None
        return b, y

    def train_order(self, bs: int, rng) -> list[np.ndarray]:
        perm = rng.permutation(len(self.ids))
        out = []
        for s in range(0, len(perm), bs * 50):  # sort within mega-batches -> little padding
            chunk = perm[s:s + bs * 50]
            chunk = chunk[np.argsort(self.lens[chunk], kind="stable")]
            out += [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
        return [out[i] for i in rng.permutation(len(out))]

    def eval_order(self, bs: int) -> list[np.ndarray]:
        order = np.argsort(self.lens, kind="stable")
        return [order[i:i + bs] for i in range(0, len(order), bs)]

    def padded_tokens(self, batches) -> int:
        return int(sum(self.lens[b].max() * len(b) for b in batches))


def autocast(cfg, device):
    import contextlib
    import torch
    if device.type == "cuda" and cfg.bf16:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def predict(model, data: Batches, cfg, device) -> np.ndarray:
    import torch
    model.eval()
    out = np.empty(len(data.ids), dtype=np.float32)
    with torch.inference_mode(), autocast(cfg, device):
        for b in data.eval_order(cfg.eval_batch_size):
            x, _ = data.collate(b)
            out[b] = model(**x).float().cpu().numpy()
    return out


# =========================================================================== metrics
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def pair_sweep(scores, labels, n_true_all, thresholds) -> list[dict]:
    """Pair-level P / R / F0.5. `recall` = of the true pairs that reached this stage;
    `recall_e2e` = of ALL ground-truth pairs (Stage-1/2 misses count as misses)."""
    npos = int(labels.sum())
    out = []
    for t in thresholds:
        sel = scores >= t
        tp = int(labels[sel].sum())
        p = tp / max(int(sel.sum()), 1)
        r = tp / max(npos, 1)
        r2 = tp / max(n_true_all, 1)
        f = 1.25 * p * r / (0.25 * p + r) if tp else 0.0
        f2 = 1.25 * p * r2 / (0.25 * p + r2) if tp else 0.0
        out.append({"threshold": t, "n_pred": int(sel.sum()), "precision": round(p, 4),
                    "recall": round(r, 4), "f05": round(f, 4), "recall_e2e": round(r2, 4),
                    "f05_e2e": round(f2, 4)})
    return out


def best_pair_f05(scores, labels) -> tuple[float, float]:
    """Max pair F0.5 over all distinct thresholds (early-stopping criterion)."""
    o = np.argsort(-scores, kind="stable")
    y = labels[o]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    p, r = tp / k, tp / max(y.sum(), 1)
    f = np.where(tp > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)
    i = int(np.argmax(f))
    return float(f[i]), float(scores[o][i])


class EntityScorer:
    """Challenge metric: macro F0.5 over EVERY S1 of the split (entities with no passed
    pairs predict empty; empty-vs-empty = 1.0), plus the singleton / empty breakdown."""

    def __init__(self, pairs: pd.DataFrame, all_s1: list[str], truth: dict):
        self.ids = list(truth)  # every S1 with ground truth
        code = pd.Series(np.arange(len(self.ids)), index=self.ids)
        self.code = code.reindex(pairs["s1"].to_numpy()).to_numpy()
        assert not np.isnan(self.code.astype(float)).any(), "pair S1 missing from truth"
        self.code = self.code.astype(np.int64)
        self.n_true = np.array([len(truth[s]) for s in self.ids], dtype=np.float64)
        self.labels = pairs["label"].to_numpy()
        self.has_pairs = np.bincount(self.code, minlength=len(self.ids)) > 0
        self.country = None

    def f05_per_entity(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        E = len(self.ids)
        tp = np.bincount(self.code, weights=(pred & (self.labels > 0)).astype(np.float64), minlength=E)
        npred = np.bincount(self.code, weights=pred.astype(np.float64), minlength=E)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = tp / npred
            r = tp / self.n_true
            f = 1.25 * p * r / (0.25 * p + r)
        f = np.where(tp > 0, f, 0.0)
        f = np.where((npred == 0) & (self.n_true == 0), 1.0, f)
        return f, npred

    def sweep(self, scores, thresholds) -> list[dict]:
        single = self.n_true == 0
        out = []
        for t in thresholds:
            f, npred = self.f05_per_entity(scores >= t)
            empty = npred == 0
            out.append({"threshold": t, "macro_f05": round(float(f.mean()), 4),
                        "pred_empty": int(empty.sum()),
                        "true_singletons": int(single.sum()),
                        "empty_correct": int((empty & single).sum()),
                        "empty_wrong": int((empty & ~single).sum()),
                        "singleton_false_merge": int((~empty & single).sum())})
        return out


def calibration(prob, labels, bins=10) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    ece = 0.0
    rel = []
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(prob[m].mean() - labels[m].mean())
            rel.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                        "mean_pred": round(float(prob[m].mean()), 3),
                        "frac_pos": round(float(labels[m].mean()), 3)})
    brier = float(np.mean((prob - labels) ** 2))
    return {"ece": round(float(ece), 4), "brier": round(brier, 4), "reliability": rel}


def fit_platt(logits, labels) -> tuple[float, float]:
    """1-D logistic regression on dev logits -> calibrated P(match) = sigmoid(a*z + b)."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(logits.reshape(-1, 1), labels)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def normalized_scores(s1: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Softmax over the S1's own candidates plus a 'no match' slot at calibrated logit 0:
    norm_i = odds_i / (1 + sum_j odds_j). The slot's share (returned second, per pair) is
    what makes 'reject all K' expressible; a lone candidate gets norm == raw."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    tot = pd.Series(odds).groupby(s1).transform("sum").to_numpy()
    return (odds / (1 + tot)).astype(np.float32), (1 / (1 + tot)).astype(np.float32)


# =========================================================================== train
def train(cfg, model, fit: Batches, dev: Batches, dev_pairs: pd.DataFrame, out_dir: str,
          device, pos_weight: float) -> dict:
    import torch
    from transformers import get_linear_schedule_with_warmup

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("pooler.", "head."))]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(("pooler.", "head."))]
    opt = torch.optim.AdamW([{"params": body, "lr": cfg.lr}, {"params": head, "lr": cfg.head_lr}],
                            weight_decay=0.01)
    steps_per_epoch = math.ceil(len(fit.ids) / cfg.batch_size)
    total = cfg.max_steps or steps_per_epoch * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup * total), total)
    eval_every = cfg.eval_every or max(steps_per_epoch // cfg.evals_per_epoch, 20)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    dev_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    dev_labels = dev_pairs["label"].to_numpy()

    hist, best, bad, step = [], {"f05": -1.0}, 0, 0
    t0, seen, run_loss, run_n = time.perf_counter(), 0, 0.0, 0
    print(f"    {total} steps ({steps_per_epoch}/epoch, bs {cfg.batch_size}), eval every "
          f"{eval_every}, patience {cfg.patience}", flush=True)
    done = False
    while not done:
        for b in fit.train_order(cfg.batch_size, rng):
            model.train()
            x, y = fit.collate(b)
            with autocast(cfg, device):
                logit = model(**x)
            loss = loss_fn(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += len(b)
            run_loss += float(loss) * len(b)
            run_n += len(b)
            if step % eval_every == 0 or step == total:
                el = time.perf_counter() - t0
                z = predict(model, dev, cfg, device)
                dl = float(dev_loss_fn(torch.from_numpy(z), torch.from_numpy(dev_labels)))
                f, thr = best_pair_f05(z, dev_labels)
                rec = {"step": step, "epoch": round(step / steps_per_epoch, 2),
                       "train_loss": round(run_loss / max(run_n, 1), 4), "dev_loss": round(dl, 4),
                       "dev_pair_f05": round(f, 4), "dev_best_logit_thr": round(thr, 3),
                       "train_pairs_per_s": round(seen / el, 1), "elapsed_s": round(el, 1)}
                hist.append(rec)
                run_loss = run_n = 0
                improved = f > best["f05"] + 1e-4
                print(f"    step {step:>6} ep {rec['epoch']:<5} train_loss {rec['train_loss']:.4f} "
                      f"dev_loss {dl:.4f} dev_pairF0.5 {f:.4f} {'*' if improved else ' '} "
                      f"| {rec['train_pairs_per_s']} pairs/s | {el / 60:.1f} min", flush=True)
                if improved:
                    best, bad = {"f05": f, "step": step}, 0
                    save_state(model, cfg, out_dir)
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        print(f"    early stop at step {step} (best {best})", flush=True)
                        done = True
                        break
            if step >= total:
                done = True
                break
    return {"history": hist, "best": best, "steps": step,
            "train_seconds": round(time.perf_counter() - t0, 1),
            "train_pairs_seen": seen}


def save_state(model, cfg, out_dir):
    """Best checkpoint: adapter weights only for LoRA, full backbone otherwise."""
    import torch
    if cfg.mode == "lora":
        model.backbone.save_pretrained(os.path.join(out_dir, "adapter"))
    else:
        model.backbone.save_pretrained(os.path.join(out_dir, "backbone"))
    torch.save(head_state(model), os.path.join(out_dir, "head.pt"))


def load_trained(model_dir: str, device):
    """Rebuild a saved cross-encoder (+ its meta) for scoring."""
    import torch
    from transformers import AutoModel
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    ns = argparse.Namespace(**meta["model_cfg"])
    if ns.mode == "lora":
        from peft import PeftModel
        ns_base = argparse.Namespace(**{**vars(ns), "mode": "none"})
        model = build_model(ns_base, len(meta["feats"]))
        model.backbone = PeftModel.from_pretrained(model.backbone, os.path.join(model_dir, "adapter"))
    else:
        model = build_model(argparse.Namespace(**{**vars(ns), "mode": "none"}), len(meta["feats"]))
        model.backbone = AutoModel.from_pretrained(os.path.join(model_dir, "backbone"),
                                                   dtype=torch.float32)
    model.load_state_dict(torch.load(os.path.join(model_dir, "head.pt")), strict=False)
    return model.to(device).eval(), meta


# =========================================================================== cost
def benchmark(cfg, model, data: Batches, device, n_train=12, n_infer=12) -> dict:
    """Measured train / inference pairs per second on THIS device at the chosen settings."""
    import torch
    rng = np.random.default_rng(0)
    order = data.train_order(cfg.batch_size, rng)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-9)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    res = {}
    model.train()
    k = min(n_train + 2, len(order))
    for i in range(k):
        if i == 2:
            sync(); t0 = time.perf_counter(); n = 0
        x, y = data.collate(order[i])
        with autocast(cfg, device):
            loss = loss_fn(model(**x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i >= 2:
            n += len(order[i])
    sync()
    res["train_pairs_per_s"] = round(n / (time.perf_counter() - t0), 1)
    model.eval()
    ev = data.eval_order(cfg.eval_batch_size)
    ev = [ev[i] for i in rng.permutation(len(ev))[:n_infer + 2]]
    with torch.inference_mode(), autocast(cfg, device):
        for i, b in enumerate(ev):
            if i == 2:
                sync(); t0 = time.perf_counter(); n = 0
            x, _ = data.collate(b)
            model(**x)
            if i >= 2:
                n += len(b)
    sync()
    res["infer_pairs_per_s"] = round(n / (time.perf_counter() - t0), 1)
    model.zero_grad(set_to_none=True)
    return res


def gpu_estimate(cfg, mean_tokens: float, n_body_params: float) -> dict:
    """Analytic A10G (g5.xlarge) throughput: FLOPs/pair = k * params * tokens with k = 6
    (full: fwd + bwd incl. weight grads), 4 (LoRA: no frozen-weight grads), 2 (inference);
    divided by peak bf16 TFLOPS x assumed model-FLOPs utilization."""
    eff = cfg.gpu_tflops * 1e12 * cfg.gpu_mfu
    k_train = 4 if cfg.mode == "lora" else 6
    return {"train_pairs_per_s": round(eff / (k_train * n_body_params * mean_tokens), 0),
            "infer_pairs_per_s": round(eff / (2 * n_body_params * mean_tokens), 0),
            "assumptions": {"peak_bf16_tflops": cfg.gpu_tflops, "mfu": cfg.gpu_mfu,
                            "non_embedding_params_M": round(n_body_params / 1e6, 1),
                            "mean_padded_tokens_per_pair": round(mean_tokens, 1)}}


def full_scale_rows(cfg, pairs_per_s1: dict) -> dict:
    """Projected full-dataset pair counts = full S1 count x pairs/S1 observed here.
    Test pairs/S1 is taken from val (same Stage-2 filter)."""
    out = {}
    full = {"train": cfg.full_train_s1, "val": cfg.full_val_s1, "test": cfg.full_test_s1}
    pairs_per_s1 = {**pairs_per_s1, "test": pairs_per_s1.get("val")}
    for split, pps in pairs_per_s1.items():
        p = full[split]
        if p and os.path.exists(p) and pps is not None:
            with open(p, "rb") as f:
                n = sum(1 for _ in f) - 1
            out[split] = {"s1": n, "pairs_per_s1": round(pps, 3), "pairs": int(n * pps)}
    return out


def cost_report(cfg, rows_here, bench, gpu, full_rows) -> dict:
    tr = full_rows.get("train", {}).get("pairs")
    if cfg.max_train_s1 and "train" in full_rows:
        tr = int(min(cfg.max_train_s1, full_rows["train"]["s1"])
                 * full_rows["train"]["pairs_per_s1"])
    inf = sum(full_rows.get(s, {}).get("pairs", 0) for s in ("val", "test"))
    rep = {"rows_this_run": rows_here, "measured_here": bench, "a10g_estimate": gpu,
           "full_scale_rows": full_rows}
    if tr:
        lo, hi = 0.5, 1.5  # throughput uncertainty band around the analytic estimate
        dev_pairs = tr * cfg.dev_frac
        if cfg.max_dev_s1:
            dev_pairs = min(dev_pairs, cfg.max_dev_s1 * full_rows["train"]["pairs_per_s1"])
        fit_pairs = (tr - dev_pairs) * cfg.epochs
        h_train = fit_pairs / gpu["train_pairs_per_s"] / 3600
        h_eval = dev_pairs * cfg.evals_per_epoch * cfg.epochs / gpu["infer_pairs_per_s"] / 3600
        h_inf = inf / gpu["infer_pairs_per_s"] / 3600
        h = h_train + h_eval + h_inf
        rep["full_run"] = {
            "train_pairs": tr, "epochs": cfg.epochs, "score_pairs_val_test": inf,
            "gpu_hours": {"train": round(h_train, 2), "dev_evals": round(h_eval, 2),
                          "score_val_test": round(h_inf, 2), "total": round(h, 2),
                          "range": [round(h / hi, 2), round(h / lo, 2)]},
            "cost_usd_spot": {"at": cfg.spot_price, "total": round(h * cfg.spot_price, 2),
                              "range": [round(h / hi * cfg.spot_price, 2),
                                        round(h / lo * cfg.spot_price, 2)]}}
    return rep


# =========================================================================== outputs
def score_split(model, sp, cfg, tok, device, max_len, feat_mu, feat_sd, platt) -> pd.DataFrame:
    pairs = sp["pairs"]
    feats = standardize(pairs, cfg.use_feats, feat_mu, feat_sd)
    data = Batches(tokenize(tok, pairs, max_len), feats, None, tok.pad_token_id, device)
    z = predict(model, data, cfg, device)
    prob = sigmoid(platt[0] * z + platt[1]).astype(np.float32)
    norm, null = normalized_scores(pairs["s1"].to_numpy(), prob)
    out = pd.DataFrame({"source1_entity_id": pairs["s1"], "candidate_entity_id": pairs["cand"],
                        "raw_score": prob, "normalized_score": norm, "raw_logit": z,
                        "no_match_score": null, "lgbm_score": pairs["lgbm_score"],
                        "lgbm_rank": pairs["lgbm_rank"].astype(np.int32)})
    if "label" in pairs:
        out["_label"] = pairs["label"].to_numpy()
    return out


def write_scores(path: str, df: pd.DataFrame):
    d = df.drop(columns=[c for c in df if c.startswith("_")])
    d = d.sort_values(["source1_entity_id", "raw_score"], ascending=[True, False], kind="stable")
    tmp = path + ".tmp"
    d.to_csv(tmp, sep="\t", index=False, float_format="%.6f", lineterminator="\n")
    os.replace(tmp, path)


def standardize(pairs, use_feats, mu=None, sd=None) -> np.ndarray:
    if not use_feats:
        return np.zeros((len(pairs), 0), np.float32)
    X = pairs[FEATS].to_numpy(np.float32)
    return ((X - np.asarray(mu, np.float32)) / np.asarray(sd, np.float32)).astype(np.float32)


def val_report(scored: pd.DataFrame, sp) -> dict:
    truth, pairs = sp["truth"], sp["pairs"]
    ent = EntityScorer(pairs, sp["all_s1"], truth)
    y = scored["_label"].to_numpy()
    n_true_all = sum(len(t) for t in truth.values())
    rep = {}
    for key in ("raw_score", "normalized_score", "lgbm_score"):
        s = scored[key].to_numpy()
        pt = pair_sweep(s, y, n_true_all, THRESHOLDS)
        et = ent.sweep(s, THRESHOLDS)
        rep[key] = [{**a, **{k: v for k, v in b.items() if k != "threshold"}} for a, b in zip(pt, et)]
    # oracle: a perfect classifier over the pairs that reached Stage 3
    f, _ = ent.f05_per_entity(y > 0)
    rep["oracle_macro_f05"] = round(float(f.mean()), 4)
    f, _ = ent.f05_per_entity(np.zeros(len(y), bool))
    rep["all_empty_macro_f05"] = round(float(f.mean()), 4)
    rep["entities"] = {"s1_total": len(truth), "s1_with_pairs": int(ent.has_pairs.sum()),
                       "s1_no_pairs_from_stage2": int((~ent.has_pairs).sum()),
                       "true_singletons": int((ent.n_true == 0).sum()),
                       "true_singletons_with_no_pairs": int(((ent.n_true == 0) & ~ent.has_pairs).sum()),
                       "true_pairs_total": int(n_true_all), "true_pairs_reaching_stage3": int(y.sum()),
                       "recall_ceiling": round(float(y.sum()) / max(n_true_all, 1), 4)}
    from sklearn.metrics import average_precision_score
    rep["average_precision"] = {k: round(float(average_precision_score(y, scored[k])), 4)
                                for k in ("raw_score", "normalized_score", "lgbm_score")}
    rep["calibration_raw_score"] = calibration(scored["raw_score"].to_numpy(), y)
    rep["calibration_uncalibrated_sigmoid"] = calibration(sigmoid(scored["raw_logit"].to_numpy()), y)
    return rep


# =========================================================================== main
def _device(cfg):
    import torch
    torch.set_num_threads(cfg.n_jobs)
    return torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def run_train(cfg):
    from transformers import AutoTokenizer

    t_start = time.perf_counter()
    require_files(cfg.train_candidates, cfg.train_s1, cfg.train_truth, cfg.val_candidates,
                  cfg.val_s1, cfg.val_truth, cfg.s2, cfg.s3)
    device = _device(cfg)
    out_dir = cfg.model_dir
    os.makedirs(out_dir, exist_ok=True)
    rep = {"config": vars(cfg).copy(), "device": str(device), "run_dir": out_dir}
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    with stage("load Stage-2 filtered pairs + texts"):
        tr = build_split("train", cfg.train_candidates, cfg.train_s1, cfg.s2, cfg.s3,
                         cfg.train_truth, cfg.n_jobs)
        va = build_split("val", cfg.val_candidates, cfg.val_s1, cfg.s2, cfg.s3,
                         cfg.val_truth, cfg.n_jobs)
        for sp in (tr, va):
            print(f"    {sp['name']}: {len(sp['all_s1']):,} S1 | {len(sp['pairs']):,} pairs | "
                  f"{sp['pairs']['s1'].nunique():,} S1 with >=1 pair", flush=True)

    # ---- class balance + grouped split -------------------------------------------------
    m_fit, m_dev = grouped_fit_dev(tr["pairs"], cfg)
    fit_p = tr["pairs"][m_fit].reset_index(drop=True)
    dev_p = tr["pairs"][m_dev].reset_index(drop=True)
    bal = {}
    for name, d in (("fit", fit_p), ("dev", dev_p), ("val", va["pairs"])):
        pos, n = int(d["label"].sum()), len(d)
        bal[name] = {"s1": int(d["s1"].nunique()), "pairs": n, "pos": pos, "neg": n - pos,
                     "pos_rate": round(pos / max(n, 1), 4),
                     "neg_per_pos": round((n - pos) / max(pos, 1), 3),
                     "mean_lgbm_score": round(float(d["lgbm_score"].mean()), 4)}
    rep["class_balance"] = bal
    print("\nCLASS BALANCE (pairs reaching Stage 3)")
    for k, v in bal.items():
        print(f"  {k:<4} S1 {v['s1']:>8,} | pairs {v['pairs']:>9,} | pos {v['pos']:>8,} | "
              f"neg {v['neg']:>8,} | pos rate {v['pos_rate']:.4f} | neg:pos {v['neg_per_pos']} | "
              f"mean LGBM score {v['mean_lgbm_score']}")
    r = bal["fit"]["neg_per_pos"]
    pos_weight = float(np.clip(r, 0.1, 10.0)) if cfg.pos_weight == "auto" else float(cfg.pos_weight)
    rep["pos_weight"] = round(pos_weight, 4)
    print(f"  BCE pos_weight = {pos_weight:.3f} (balanced; Platt scaling on dev restores "
          f"calibrated probabilities)")

    # ---- sequence length ---------------------------------------------------------------
    with stage("token-length distribution"):
        ls = length_stats(tok, fit_p, cfg.length_sample, cfg.seed)
        max_len = choose_max_len(ls, cfg)
        lens = ls.pop("_pair_lengths")
        ls["chosen_max_len"] = max_len
        ls["truncated_pct"] = round(100 * float((lens > max_len).mean()), 3)
        rep["token_lengths"] = ls
        print(f"    pair tokens (untruncated, incl. specials): {ls['pair']} | one side: "
              f"{ls['one_side']}\n    -> max_len {max_len} ({ls['truncated_pct']}% of pairs truncated)")

    with stage("tokenize"):
        feat_mu = fit_p[FEATS].mean().tolist() if cfg.use_feats else []
        feat_sd = (fit_p[FEATS].std().replace(0, 1).fillna(1).tolist()) if cfg.use_feats else []
        mk = lambda p: Batches(tokenize(tok, p, max_len),  # noqa: E731
                               standardize(p, cfg.use_feats, feat_mu, feat_sd),
                               p["label"].to_numpy(np.float32), tok.pad_token_id, device)
        fit, dev = mk(fit_p), mk(dev_p)

    model = build_model(cfg, len(FEATS) if cfg.use_feats else 0).to(device)
    rep["params"] = count_params(model)
    print(f"    model: {rep['params']}", flush=True)

    # ---- cost check ----------------------------------------------------------------------
    with stage("throughput benchmark"):
        bench = benchmark(cfg, model, fit, device)
        train_batches = fit.train_order(cfg.batch_size, np.random.default_rng(0))
        mean_tok = fit.padded_tokens(train_batches) / len(fit.ids)
        n_body = sum(p.numel() for n, p in model.named_parameters()
                     if "embeddings" not in n and "lora_" not in n)
        gpu = gpu_estimate(cfg, mean_tok, n_body)
        pps = {"train": len(tr["pairs"]) / max(len(tr["all_s1"]), 1),
               "val": len(va["pairs"]) / max(len(va["all_s1"]), 1)}
        cost = cost_report(cfg, {"fit": len(fit_p), "dev": len(dev_p), "val": len(va["pairs"])},
                           bench, gpu, full_scale_rows(cfg, pps))
        cpu_min = (len(fit_p) * cfg.epochs / max(bench["train_pairs_per_s"], 1e-9)) / 60
        cost["this_run_estimate_min"] = round(cpu_min, 1)
        rep["cost"] = cost
    print_cost(cost, cfg)
    with open(os.path.join(out_dir, "stage3_cost_estimate.json"), "w") as f:
        json.dump(cost, f, indent=2, default=float)
    if cfg.estimate_only:
        return rep

    # ---- fine-tune -----------------------------------------------------------------------
    del model
    model = build_model(cfg, len(FEATS) if cfg.use_feats else 0).to(device)  # fresh weights
    model_cfg = {k: getattr(cfg, k) for k in ("model_name", "mode", "lora_r", "lora_alpha",
                                              "lora_dropout", "gradient_checkpointing", "bf16")}
    meta = {"model_cfg": model_cfg, "feats": FEATS if cfg.use_feats else [], "feat_mu": feat_mu,
            "feat_sd": feat_sd, "max_len": max_len, "pos_weight": pos_weight}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    with stage(f"fine-tune ({cfg.mode})"):
        tr_res = train(cfg, model, fit, dev, dev_p, out_dir, device, pos_weight)
    rep["training"] = tr_res
    model, meta = load_trained(out_dir, device)  # best checkpoint

    # ---- calibrate on dev, evaluate on val -------------------------------------------------
    with stage("calibrate (dev) + score val"):
        z_dev = predict(model, dev, cfg, device)
        platt = fit_platt(z_dev, dev_p["label"].to_numpy())
        f_dev, t_dev = best_pair_f05(z_dev, dev_p["label"].to_numpy())
        meta.update({"platt_a": platt[0], "platt_b": platt[1],
                     "dev_best_pair_f05": round(f_dev, 4),
                     "dev_best_raw_threshold": round(float(sigmoid(platt[0] * t_dev + platt[1])), 4)})
        json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
        cfg.use_feats = bool(meta["feats"])
        val_sc = score_split(model, va, cfg, tok, device, max_len, feat_mu, feat_sd, platt)
        write_scores(ensure_parent(cfg.val_output), val_sc)
        rep["val"] = val_report(val_sc, va)
        print(f"    wrote {cfg.val_output}", flush=True)
    rep["platt"] = {"a": round(platt[0], 4), "b": round(platt[1], 4)}
    rep["dev_best"] = {"pair_f05": meta["dev_best_pair_f05"],
                       "raw_threshold": meta["dev_best_raw_threshold"]}
    rep["runtime_s"] = round(time.perf_counter() - t_start, 1)
    sweep = pd.DataFrame([{"score": k, **r} for k in ("raw_score", "normalized_score", "lgbm_score")
                          for r in rep["val"][k]])
    sweep.to_csv(os.path.join(out_dir, "stage3_val_sweep.tsv"), sep="\t", index=False)
    with open(os.path.join(out_dir, "stage3_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print_report(rep, cfg)
    return rep


def run_score(cfg):
    """Score one Stage-2 filtered candidates file with a trained model dir."""
    from transformers import AutoTokenizer
    t0 = time.perf_counter()
    require_files(os.path.join(cfg.model_dir, "meta.json"), os.path.join(cfg.model_dir, "head.pt"))
    device = _device(cfg)
    with stage("load Stage-2 filtered pairs + texts"):
        sp = build_split("scored", cfg.candidates, cfg.s1, cfg.s2, cfg.s3, cfg.truth, cfg.n_jobs)
        print(f"    {len(sp['all_s1']):,} S1 | {len(sp['pairs']):,} pairs", flush=True)
    model, meta = load_trained(cfg.model_dir, device)
    if "platt_a" not in meta:
        raise ValueError(f"{cfg.model_dir}/meta.json has no calibration: training did not finish")
    tok = AutoTokenizer.from_pretrained(meta["model_cfg"]["model_name"])
    cfg.use_feats = bool(meta["feats"])
    with stage(f"score {len(sp['pairs']):,} pairs"):
        sc = score_split(model, sp, cfg, tok, device, meta["max_len"], meta["feat_mu"],
                         meta["feat_sd"], (meta["platt_a"], meta["platt_b"]))
        write_scores(ensure_parent(cfg.output), sc)
        print(f"    wrote {len(sc):,} pairs -> {cfg.output}", flush=True)
    rep = {"candidates": cfg.candidates, "pairs": len(sc),
           "s1_with_pairs": int(sc["source1_entity_id"].nunique()),
           "mean_raw_score": round(float(sc["raw_score"].mean()), 4) if len(sc) else None,
           "runtime_s": round(time.perf_counter() - t0, 1)}
    if sp["truth"] is not None:
        rep["eval"] = val_report(sc, sp)
    if cfg.report:
        with open(ensure_parent(cfg.report), "w") as f:
            json.dump(rep, f, indent=2, default=float)
    return rep


# =========================================================================== printing
def print_cost(c, cfg):
    print("\n" + "=" * 78 + "\nCOST CHECK\n" + "=" * 78)
    print(f"rows this run: {c['rows_this_run']}")
    print(f"measured on this device: {c['measured_here']}  -> this run's training ~"
          f"{c['this_run_estimate_min']} min for {cfg.epochs} epoch(s) (before early stopping)")
    g = c["a10g_estimate"]
    print(f"A10G analytic estimate: train {g['train_pairs_per_s']:.0f} pairs/s | infer "
          f"{g['infer_pairs_per_s']:.0f} pairs/s | {g['assumptions']}")
    for s, v in c["full_scale_rows"].items():
        print(f"  full-scale {s:<12} {v['s1']:>9,} S1 x {v['pairs_per_s1']} pairs/S1 = {v['pairs']:>10,} pairs")
    if "full_run" in c:
        fr = c["full_run"]
        h, d = fr["gpu_hours"], fr["cost_usd_spot"]
        print(f"FULL RUN ({cfg.mode}, {fr['train_pairs']:,} train pairs x {fr['epochs']} epoch(s), "
              f"score {fr['score_pairs_val_test']:,} val+test pairs):")
        print(f"  GPU-hours {h['total']} (train {h['train']}, dev evals {h['dev_evals']}, "
              f"scoring {h['score_val_test']}; range {h['range'][0]}-{h['range'][1]})")
        print(f"  g5.xlarge spot @ ${d['at']}/h: ${d['total']} (range ${d['range'][0]}-${d['range'][1]})")
    print("=" * 78, flush=True)


def print_report(rep, cfg):
    v = rep["val"]
    print("\n" + "=" * 78 + f"\nSTAGE 3 CROSS-ENCODER ({cfg.mode}{'' if cfg.use_feats else ', text only'}) "
          "SUMMARY\n" + "=" * 78)
    print(f"params {rep['params']} | best dev pair F0.5 {rep['dev_best']['pair_f05']} "
          f"@ raw {rep['dev_best']['raw_threshold']} (step {rep['training']['best'].get('step')}) | "
          f"train {rep['training']['train_seconds']}s")
    print(f"val: {v['entities']}")
    print(f"val average precision: {v['average_precision']}")
    cal, unc = v["calibration_raw_score"], v["calibration_uncalibrated_sigmoid"]
    print(f"val calibration raw_score: ECE {cal['ece']} Brier {cal['brier']} "
          f"(before Platt: ECE {unc['ece']} Brier {unc['brier']})")
    for key in ("raw_score", "normalized_score", "lgbm_score"):
        tag = {"lgbm_score": "   <- Stage-2 score as the decision rule (baseline)"}.get(key, "")
        print(f"\nVAL THRESHOLD SWEEP on {key}{tag}")
        print(f"  {'thr':>5} {'n_pred':>7} {'prec':>6} {'rec':>6} {'F0.5':>6} {'rec_e2e':>7} "
              f"{'F0.5_e2e':>8} {'macroF0.5':>9} {'predEmpty':>9} {'emptyOK':>7} {'emptyWrong':>10} "
              f"{'singletonFP':>11}")
        for r in v[key]:
            print(f"  {r['threshold']:>5} {r['n_pred']:>7,} {r['precision']:>6.4f} {r['recall']:>6.4f} "
                  f"{r['f05']:>6.4f} {r['recall_e2e']:>7.4f} {r['f05_e2e']:>8.4f} {r['macro_f05']:>9.4f} "
                  f"{r['pred_empty']:>9,} {r['empty_correct']:>7,} {r['empty_wrong']:>10,} "
                  f"{r['singleton_false_merge']:>11,}")
    print(f"\n  true singletons in val: {v['entities']['true_singletons']:,} | oracle macro F0.5 "
          f"(perfect Stage 3) {v['oracle_macro_f05']} | predict-all-empty {v['all_empty_macro_f05']}")
    print(f"runtime {rep['runtime_s']}s | outputs in {rep['run_dir']}\n" + "=" * 78, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-jobs", type=int, default=default_n_jobs())
    common.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    common.add_argument("--eval-batch-size", type=int, default=128)
    common.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                        help="CUDA only")
    pool = common.add_argument_group("pool inputs (the S2/S3 files Stage 1 matched against)")
    pool.add_argument("--s2", required=True, help="Source-2 TSV")
    pool.add_argument("--s3", required=True, help="Source-3 TSV")

    t = sub.add_parser("train", parents=[common], help="fine-tune + calibrate the cross-encoder")
    i = t.add_argument_group("inputs")
    i.add_argument("--train-candidates", required=True,
                   help="Stage-2 OUT-OF-FOLD filtered candidates TSV for train_split")
    i.add_argument("--train-s1", required=True, help="Source-1 TSV of --train-candidates")
    i.add_argument("--train-truth", required=True, help="ground truth for --train-s1")
    i.add_argument("--val-candidates", required=True, help="Stage-2 filtered candidates TSV for val")
    i.add_argument("--val-s1", required=True, help="Source-1 TSV of --val-candidates")
    i.add_argument("--val-truth", required=True, help="ground truth for --val-s1")
    o = t.add_argument_group("outputs")
    o.add_argument("--model-dir", required=True, help="model weights + meta + reports directory")
    o.add_argument("--val-output", help="val pair scores TSV (Stage-4 tuning input; "
                                        "required unless --estimate-only)")
    c = t.add_argument_group("cost projection (optional full-size S1 files, row counts only)")
    c.add_argument("--full-train-s1")
    c.add_argument("--full-val-s1")
    c.add_argument("--full-test-s1")
    t.add_argument("--seed", type=int, default=0)
    # data
    t.add_argument("--max-train-s1", type=int, default=0, help="subsample training S1 (0 = all)")
    t.add_argument("--dev-frac", type=float, default=0.15)
    t.add_argument("--max-dev-s1", type=int, default=30_000)
    t.add_argument("--max-len", type=int, default=0, help="0 = p99.5 of pair token lengths")
    t.add_argument("--length-sample", type=int, default=100_000)
    t.add_argument("--use-feats", action=argparse.BooleanOptionalAction, default=True,
                   help="fuse Stage-2 context features at the head")
    # model / training
    t.add_argument("--model-name", default="microsoft/deberta-v3-base")
    t.add_argument("--mode", choices=["lora", "full"], default="lora")
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--lora-alpha", type=int, default=32)
    t.add_argument("--lora-dropout", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=None, help="default 2e-4 (lora) / 2e-5 (full)")
    t.add_argument("--head-lr", type=float, default=1e-3)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--warmup", type=float, default=0.06)
    t.add_argument("--evals-per-epoch", type=int, default=4)
    t.add_argument("--eval-every", type=int, default=0)
    t.add_argument("--patience", type=int, default=4, help="evals without dev F0.5 gain")
    t.add_argument("--pos-weight", default="auto", help="'auto' = fit neg/pos (clipped), or a number")
    t.add_argument("--gradient-checkpointing", action="store_true")
    # cost check
    t.add_argument("--estimate-only", action="store_true")
    t.add_argument("--gpu-tflops", type=float, default=70.0, help="A10G peak bf16 dense (assumed)")
    t.add_argument("--gpu-mfu", type=float, default=0.25, help="assumed utilization")
    t.add_argument("--spot-price", type=float, default=0.45, help="g5.xlarge spot $/h (assumed)")

    r = sub.add_parser("score", parents=[common], help="score one filtered candidates file")
    i = r.add_argument_group("inputs")
    i.add_argument("--model-dir", required=True, help="directory written by train")
    i.add_argument("--candidates", required=True, help="Stage-2 filtered candidates TSV")
    i.add_argument("--s1", required=True, help="Source-1 TSV of --candidates")
    i.add_argument("--truth", help="ground truth for --s1 (optional: adds evaluation)")
    o = r.add_argument_group("outputs")
    o.add_argument("--output", required=True, help="pair scores TSV")
    o.add_argument("--report", help="report JSON")
    cfg = p.parse_args(argv)
    if cfg.cmd == "train":
        if cfg.lr is None:
            cfg.lr = 2e-4 if cfg.mode == "lora" else 2e-5
        if not cfg.estimate_only and not cfg.val_output:
            p.error("train: --val-output is required unless --estimate-only")
    return cfg


if __name__ == "__main__":
    args = parse_args()
    (run_train if args.cmd == "train" else run_score)(args)
