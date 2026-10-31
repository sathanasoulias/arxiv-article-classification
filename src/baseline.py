"""Classical baselines for predicting the discovered categories.

A fine-tuned transformer is only worth its cost if it beats something cheap. These
baselines establish that reference point, and they also act as a lower bound on how
learnable the discovered taxonomy is: if a linear model on bag-of-words already
separates the clusters well, the categories are at least lexically coherent.

Three baselines, in increasing order of what they know:
  majority  — always predicts the largest cluster. Shows how misleading accuracy is
              under imbalance, and why macro-F1 is the headline metric.
  tfidf+svm — word and character n-grams into a linear SVM. The standard strong
              baseline for topical text classification.
  embed+lr  — logistic regression on the frozen sentence embeddings. This one is
              diagnostic rather than competitive: the clusters were *defined* in this
              space, so a high score here measures how linearly separable the
              clustering is, not how well the task generalises from raw text.

Input:   WORK_DIR/<phase>/sample.parquet, labels_<model>_<config>.npy,
         embeddings_<model>.npy
Outputs: reports/baselines_<phase>_<model>_<config>.json
Usage:   python baseline.py --phase pilot --labels hdbscan_mcs100
"""

import argparse
import json
import time

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

from config import REPORTS, SEED, work_path
from train_classifier import load_data


def score(name, y_true, y_pred, seconds):
    out = {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "fit_seconds": round(seconds, 1),
    }
    print(f"  {name:12} macro-F1={out['macro_f1']:.4f}  "
          f"accuracy={out['accuracy']:.4f}  ({seconds:.0f}s)")
    return out


def tfidf_svm():
    """Word n-grams catch topical vocabulary; char n-grams catch LaTeX and morphology."""
    words = TfidfVectorizer(
        sublinear_tf=True, min_df=3, ngram_range=(1, 2),
        stop_words="english", max_features=300_000,
    )
    chars = TfidfVectorizer(
        sublinear_tf=True, min_df=3, analyzer="char_wb", ngram_range=(3, 5),
        max_features=200_000,
    )
    # class_weight balanced for the same reason the transformer uses a weighted loss.
    return make_pipeline(
        make_union(words, chars),
        LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, dual="auto"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--embed-model", choices=["minilm", "mpnet"], default="minilm")
    ap.add_argument("--labels", default="hdbscan_mcs100")
    ap.add_argument("--noise", choices=["drop", "class"], default="drop")
    args = ap.parse_args()

    df, classes, n_noise = load_data(
        args.phase, args.embed_model, args.labels, args.noise)
    train, val, test = (df[df.split == s] for s in ("train", "val", "test"))
    names = [f"cluster_{c}" for c in classes]

    print(f"\n[{args.phase}/{args.embed_model}/{args.labels}] "
          f"{len(classes)} categories, train={len(train):,} val={len(val):,} "
          f"test={len(test):,}")

    results = {}

    # --- majority class ---------------------------------------------------------
    t = time.time()
    dummy = DummyClassifier(strategy="most_frequent").fit(
        train["abstract"], train["y"])
    results["majority"] = score(
        "majority", val["y"], dummy.predict(val["abstract"]), time.time() - t)

    # --- tf-idf + linear svm ----------------------------------------------------
    t = time.time()
    svm = tfidf_svm().fit(train["abstract"], train["y"])
    pred_val = svm.predict(val["abstract"])
    results["tfidf_svm"] = score("tfidf+svm", val["y"], pred_val, time.time() - t)
    # The strongest cheap model is also scored on test, as the figure the
    # transformer has to beat.
    results["tfidf_svm"]["test"] = {
        "macro_f1": round(f1_score(test["y"], svm.predict(test["abstract"]),
                                   average="macro", zero_division=0), 4),
    }
    print("\n  tf-idf + SVM, per category (validation):")
    print(classification_report(val["y"], pred_val, target_names=names,
                                zero_division=0))

    # --- logistic regression on the frozen embeddings ---------------------------
    vectors = np.load(work_path(args.phase, f"embeddings_{args.embed_model}.npy"))
    # load_data may have dropped noise rows, so realign by the original row order.
    idx = df.index.to_numpy() if len(df) == len(vectors) else None
    if idx is None:
        import pandas as pd
        full = pd.read_parquet(work_path(args.phase, "sample.parquet"),
                               columns=["id"])
        pos = {paper_id: i for i, paper_id in enumerate(full["id"])}
        idx = df["id"].map(pos).to_numpy()
    X = vectors[idx]
    mask = {s: (df["split"] == s).to_numpy() for s in ("train", "val", "test")}

    t = time.time()
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=SEED).fit(X[mask["train"]], train["y"])
    results["embed_logreg"] = score(
        "embed+logreg", val["y"], lr.predict(X[mask["val"]]), time.time() - t)
    results["embed_logreg"]["note"] = (
        "diagnostic only — the clusters were defined in this embedding space, so "
        "this measures their linear separability, not generalisation from raw text"
    )

    summary = {
        "phase": args.phase,
        "embed_model": args.embed_model,
        "label_source": f"labels_{args.embed_model}_{args.labels}.npy",
        "n_classes": len(classes),
        "noise_handling": args.noise,
        "n_noise_abstracts": n_noise,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "seed": SEED,
        "metric_note": (
            "macro-F1 is the headline metric; accuracy is reported only to show how "
            "far the majority baseline gets without learning anything"
        ),
        "baselines": results,
    }
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"baselines_{args.phase}_{args.embed_model}_{args.labels}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out.name}")


if __name__ == "__main__":
    main()
