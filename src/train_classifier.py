"""Fine-tune a BERT encoder to predict the discovered categories from an abstract.

What the targets are, and why
-----------------------------
The targets are the cluster assignments from interpret.py, NOT arXiv's own labels.
This follows from the brief: the number of categories is unknown, so the categories
are discovered first and the classifier then learns to assign a new abstract to one of
them. A model trained on arXiv's labels would be answering a different question.

Consequence for evaluation: the reported macro-F1 measures how learnable the
discovered taxonomy is, and is only comparable between runs that share the same label
set. It is not a score for the taxonomy itself — a trivial 2-cluster split would score
higher while being useless. Taxonomy quality is assessed separately, by stability.py
and interpret.py.

Class imbalance is handled with weights in the loss rather than by resampling, so
every abstract is still seen exactly once per epoch and the sampling distribution
established in preprocess.py is left intact. Macro-F1 is the headline metric because
accuracy is dominated by the largest cluster (43% of the corpus in the pilot).

Input:   WORK_DIR/<phase>/sample.parquet, labels_<model>_<config>.npy
Outputs: WORK_DIR/<phase>/clf_<tag>/            (checkpoints; not committed)
         reports/classifier_<tag>.json          (metrics + full run configuration)
Usage:   python train_classifier.py --phase pilot --labels hdbscan_mcs100
         python train_classifier.py --phase main --labels kmeans_k7 --epochs 3
"""

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
)

from config import MAX_TOKENS, REPORTS, SEED, work_path

# The brief links to HuggingFace filtered on `search=bert-base`, so bert-base-uncased
# is the default. The alternatives are compared in tune.py.
ENCODERS = {
    "bert-base": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "scibert": "allenai/scibert_scivocab_uncased",
}


class AbstractDataset(torch.utils.data.Dataset):
    """Pre-tokenised abstracts. Padding is left to the collator, per batch."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(texts, truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def load_data(phase, embed_model, config, noise):
    """Join abstracts to cluster labels and apply the train/val/test split."""
    df = pd.read_parquet(
        work_path(phase, "sample.parquet"),
        columns=["id", "abstract", "split", "primary_archive"],
    )
    label_file = work_path(phase, f"labels_{embed_model}_{config}.npy")
    if not label_file.exists():
        raise FileNotFoundError(f"{label_file} missing — run interpret.py first")
    df["cluster"] = np.load(label_file)
    assert len(df) == len(df["cluster"])

    n_noise = int((df["cluster"] == -1).sum())
    if noise == "drop":
        # An unassigned abstract carries no target, so it cannot supervise anything.
        # Dropping is honest; the cost is that coverage is below 100% and is reported.
        df = df[df["cluster"] >= 0].reset_index(drop=True)
    else:
        # Noise as its own class asks the model to learn "none of the above", which is
        # a density property of the embedding space rather than a topic.
        df["cluster"] = df["cluster"].replace(-1, df["cluster"].max() + 1)

    # Remap to contiguous 0..C-1 as the classification head requires.
    classes = sorted(df["cluster"].unique())
    remap = {c: i for i, c in enumerate(classes)}
    df["y"] = df["cluster"].map(remap)
    return df, classes, n_noise


def class_weights(y, n_classes, device):
    """Inverse-frequency weights, mean-normalised so the loss scale is unchanged."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = len(y) / (n_classes * np.maximum(counts, 1))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def make_loss(weights):
    """Weighted cross-entropy via Trainer's compute_loss_func hook."""
    def compute(outputs, labels, num_items_in_batch=None):
        return F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
            weight=weights.to(outputs.logits.device),
        )
    return compute


def metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


def evaluate(trainer, dataset, classes, split_name):
    out = trainer.predict(dataset)
    preds = np.argmax(out.predictions, axis=-1)
    labels = out.label_ids
    names = [f"cluster_{c}" for c in classes]
    report = classification_report(
        labels, preds, target_names=names, output_dict=True, zero_division=0)
    print(f"\n{split_name}: macro-F1={report['macro avg']['f1-score']:.4f}  "
          f"accuracy={report['accuracy']:.4f}")
    print(classification_report(labels, preds, target_names=names, zero_division=0))
    return {
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
        "per_class": {
            n: {k: round(v, 4) for k, v in report[n].items()} for n in names
        },
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--embed-model", choices=["minilm", "mpnet"], default="minilm")
    ap.add_argument("--labels", default="hdbscan_mcs100",
                    help="cluster config, e.g. hdbscan_mcs100 or kmeans_k7")
    ap.add_argument("--encoder", choices=list(ENCODERS), default="bert-base")
    ap.add_argument("--noise", choices=["drop", "class"], default="drop")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-len", type=int, default=MAX_TOKENS)
    ap.add_argument("--weighted-loss", action="store_true", default=True)
    ap.add_argument("--no-weighted-loss", dest="weighted_loss",
                    action="store_false")
    ap.add_argument("--max-train", type=int, default=None,
                    help="cap training rows, for CPU smoke tests only")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    tag = args.tag or (f"{args.phase}_{args.encoder}_{args.labels}"
                       f"_lr{args.lr:g}_bs{args.batch_size}_ep{args.epochs:g}"
                       f"_seed{args.seed}")
    set_seed(args.seed)

    df, classes, n_noise = load_data(
        args.phase, args.embed_model, args.labels, args.noise)
    n_classes = len(classes)

    train = df[df.split == "train"]
    if args.max_train and len(train) > args.max_train:
        train = train.sample(args.max_train, random_state=args.seed)
    val = df[df.split == "val"]
    test = df[df.split == "test"]

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[{tag}]")
    print(f"  encoder   {ENCODERS[args.encoder]}")
    print(f"  targets   {n_classes} discovered categories from {args.labels}")
    print(f"  rows      train={len(train):,} val={len(val):,} test={len(test):,}"
          f"  (noise {args.noise}: {n_noise:,} abstracts)")
    print(f"  device    {device}")

    tokeniser = AutoTokenizer.from_pretrained(ENCODERS[args.encoder])
    model = AutoModelForSequenceClassification.from_pretrained(
        ENCODERS[args.encoder],
        num_labels=n_classes,
        id2label={i: f"cluster_{c}" for i, c in enumerate(classes)},
        label2id={f"cluster_{c}": i for i, c in enumerate(classes)},
    )

    sets = {
        name: AbstractDataset(part["abstract"].tolist(), part["y"].to_numpy(),
                              tokeniser, args.max_len)
        for name, part in (("train", train), ("val", val), ("test", test))
    }

    out_dir = work_path(args.phase, f"clf_{tag}")
    targs = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(32, args.batch_size * 2),
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=int(0.1 * len(train) * args.epochs / args.batch_size),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",   # not accuracy: clusters are imbalanced
        greater_is_better=True,
        logging_steps=100,
        fp16=(device == "cuda"),
        seed=args.seed,
        report_to=[],
    )

    weights = class_weights(train["y"].to_numpy(), n_classes, device)
    print(f"  weights   {np.round(weights.cpu().numpy(), 2).tolist()}"
          if args.weighted_loss else "  weights   uniform")

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=sets["train"],
        eval_dataset=sets["val"],
        processing_class=tokeniser,
        data_collator=DataCollatorWithPadding(tokeniser),
        compute_metrics=metrics_fn,
        compute_loss_func=make_loss(weights) if args.weighted_loss else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60

    # Model selection used validation only; test is read once, at the end.
    results = {
        "val": evaluate(trainer, sets["val"], classes, "VALIDATION"),
        "test": evaluate(trainer, sets["test"], classes, "TEST"),
    }

    summary = {
        "tag": tag,
        "phase": args.phase,
        "encoder": ENCODERS[args.encoder],
        "label_source": f"labels_{args.embed_model}_{args.labels}.npy",
        "n_classes": n_classes,
        "cluster_ids": [int(c) for c in classes],
        "noise_handling": args.noise,
        "n_noise_abstracts": n_noise,
        "coverage_pct": round(100 * len(df) / (len(df) + (
            n_noise if args.noise == "drop" else 0)), 2),
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "hyperparameters": {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_length": args.max_len,
            "weight_decay": 0.01,
            "warmup_steps": targs.warmup_steps,
            "weighted_loss": args.weighted_loss,
            "early_stopping_patience": 2,
            "seed": args.seed,
        },
        "train_minutes": round(minutes, 1),
        "device": device,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "results": results,
    }
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"classifier_{tag}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nTrained in {minutes:.1f} min. Wrote {out.name}")


if __name__ == "__main__":
    main()
