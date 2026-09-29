"""Train, compare and save the sentiment pipeline.

    python -m src.train                       # uses data/sample_reviews.csv
    python -m src.train --data path/to.csv --text-col review --label-col sentiment
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import config as C
from .data import describe, load_dataset
from .preprocess import TextCleaner


def build_pipeline(classifier):
    # The vectorizer lives inside the pipeline, so it is only ever fitted on training data.
    return Pipeline([
        ("clean", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9,
                                  max_features=20000, sublinear_tf=True)),
        ("clf", classifier),
    ])


def candidates():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        # LinearSVC has no probabilities, so it is wrapped to give a confidence score
        "Linear SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
    }


def metrics_row(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": p,
            "recall_macro": r, "f1_macro": f}


def main(args):
    df = load_dataset(args.data, args.text_col, args.label_col)
    info = describe(df)
    print("Dataset:", json.dumps(info, indent=2))

    # 1. split FIRST, then fit anything
    X_train, X_test, y_train, y_test = train_test_split(
        df[C.TEXT_COL], df[C.LABEL_COL], test_size=C.TEST_SIZE,
        random_state=C.RANDOM_STATE, stratify=df[C.LABEL_COL])
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # 2. compare models: 5-fold CV on the training set only (macro F1)
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    rows = []
    for name, clf in candidates().items():
        pipe = build_pipeline(clf)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0
        row = {"model": name, "cv_f1_macro": cv_f1.mean(), "train_seconds": train_time,
               **metrics_row(y_test, pipe.predict(X_test))}
        rows.append(row)
        print(f"{name:20s} CV F1 {row['cv_f1_macro']:.3f} | test acc {row['accuracy']:.3f} "
              f"| test macro-F1 {row['f1_macro']:.3f}")
    comparison = pd.DataFrame(rows).round(4)
    comparison.to_csv(C.COMPARISON_PATH, index=False)

    # 3. pick the winner using CV score (never the test set), then tune it lightly
    best_name = comparison.sort_values("cv_f1_macro", ascending=False).iloc[0]["model"]
    print("Best by CV macro-F1:", best_name)
    grids = {
        "Naive Bayes": {"clf__alpha": [0.1, 0.5, 1.0]},
        "Logistic Regression": {"clf__C": [0.5, 1, 5, 10]},
        "Linear SVM": {"clf__estimator__C": [0.1, 0.5, 1, 2]},
    }
    grid = GridSearchCV(build_pipeline(candidates()[best_name]),
                        {"tfidf__ngram_range": [(1, 1), (1, 2)], **grids[best_name]},
                        cv=cv, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    final = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # 4. touch the test set once, for the final numbers
    y_pred = final.predict(X_test)
    final_metrics = metrics_row(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=C.LABELS, digits=3)
    print("\n" + report)

    cm = confusion_matrix(y_test, y_pred, labels=C.LABELS)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=C.LABELS, yticklabels=C.LABELS)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix: {best_name}")
    plt.tight_layout(); plt.savefig(C.CONFUSION_PATH, dpi=150); plt.close()

    # inference speed for one review
    sample = X_test.iloc[0]
    t0 = time.time()
    for _ in range(100):
        final.predict_proba([sample])
    ms = (time.time() - t0) / 100 * 1000

    C.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, C.MODEL_PATH, compress=3)
    C.REPORT_PATH.write_text(report)
    C.METRICS_PATH.write_text(json.dumps({
        "dataset": info, "train_size": len(X_train), "test_size": len(X_test),
        "split": f"{int((1 - C.TEST_SIZE) * 100)}/{int(C.TEST_SIZE * 100)} stratified, "
                 f"random_state={C.RANDOM_STATE}",
        "best_model": best_name, "best_params": {k: str(v) for k, v in grid.best_params_.items()},
        "cv_f1_macro_best": round(grid.best_score_, 4),
        "test_metrics": {k: round(v, 4) for k, v in final_metrics.items()},
        "inference_ms_per_review": round(ms, 2),
        "tfidf_max_features": 20000,
    }, indent=2))
    print(f"\nSaved model -> {C.MODEL_PATH}\nInference: {ms:.1f} ms per review")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=C.DATA_PATH)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--label-col", default=None)
    main(ap.parse_args())
