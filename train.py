"""
Train a Decision Tree that predicts an athlete's sport from their biography.

Dataset: "DBPedia Classes" (Kaggle, by danofer) - Wikipedia article abstracts
with three levels of DBpedia labels (l1, l2, l3). Rows whose l2 label is
"Athlete" are athlete biographies, and their l3 label names the sport
(e.g. "TennisPlayer"). See README.md for download instructions.

Run from the project folder:
    python train.py                 # use the CSV files in data/ (download if missing)
    python train.py --sample 3000   # quick test on a small stratified sample
    python train.py --help          # all options (other dataset, column names, ...)

Outputs:
    models/athlete_classifier.joblib   the trained pipeline (TF-IDF + Decision Tree)
    reports/model_metrics.txt          readable metrics and classification report
    reports/model_metrics.json         the same results for the dashboard
    reports/confusion_matrix.png       confusion matrix image
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # save charts without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "athlete_classifier.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_TXT = REPORTS_DIR / "model_metrics.txt"
METRICS_JSON = REPORTS_DIR / "model_metrics.json"
CONFUSION_PNG = REPORTS_DIR / "confusion_matrix.png"

# ---------------------------------------------------------------------------
# Dataset settings - change these (or use the command-line options) to train
# on a different dataset.
# ---------------------------------------------------------------------------
DATASET_NAME = "DBPedia Classes"
DATASET_SOURCE = "https://www.kaggle.com/datasets/danofer/dbpedia-classes"
DATASET_MIRROR = "https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes"  # public copy, CC0 licence
DATASET_FILES = ["DBPEDIA_train.csv", "DBPEDIA_val.csv", "DBPEDIA_test.csv"]
DOWNLOAD_FILE = "DBPEDIA_train.csv"  # the largest split (~160 MB); enough athletes on its own
DOWNLOAD_URL = f"{DATASET_MIRROR}/resolve/main/{DOWNLOAD_FILE}"

TEXT_COLUMN = "text"      # the Wikipedia abstract (biography)
LABEL_COLUMN = "l3"       # detailed class, e.g. "TennisPlayer"
FILTER_COLUMN = "l2"      # keep only rows where l2 == "Athlete"
FILTER_VALUE = "Athlete"

# DBpedia athlete classes -> project categories. Every other athlete class
# (golf, rugby, swimming, ...) becomes "Other sports".
SPORT_MAP: Dict[str, str] = {
    "SoccerPlayer": "Football",       # association football
    "Cricketer": "Cricket",
    "BasketballPlayer": "Basketball",
    "TennisPlayer": "Tennis",
    "AthleticsPlayer": "Athletics",   # track and field (used only if present in the data)
}
OTHER_LABEL = "Other sports"
CATEGORY_ORDER = ["Football", "Cricket", "Basketball", "Tennis", "Athletics", OTHER_LABEL]

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_WORDS = 5             # biographies shorter than this are dropped
TFIDF_PARAMS = dict(
    lowercase=True,
    strip_accents="unicode",
    token_pattern=r"(?u)\b[a-z]{2,}\b",  # words of 2+ letters (numbers and punctuation are ignored)
    stop_words="english",
    min_df=3,                # ignore words that appear in fewer than 3 biographies (mostly names)
    max_features=5000,
    sublinear_tf=True,
)
TREE_PARAMS = dict(
    criterion="gini",
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",  # give the smaller sports the same total weight as "Other sports"
    random_state=RANDOM_STATE,
)


class DatasetError(Exception):
    """The dataset is missing or not usable; the message explains what to do."""


def manual_download_help() -> str:
    return (
        "How to get the dataset manually:\n"
        f"  Option A (Kaggle, needs a free account): open {DATASET_SOURCE}, click 'Download',\n"
        "            unzip it, and copy DBPEDIA_train.csv (and optionally DBPEDIA_val.csv,\n"
        "            DBPEDIA_test.csv) into the project's data/ folder.\n"
        f"  Option B (no account): open {DATASET_MIRROR}/tree/main,\n"
        "            download DBPEDIA_train.csv and put it in the data/ folder.\n"
        "Then run: python train.py"
    )


# ---------------------------------------------------------------------------
# 1-2. Load and inspect
# ---------------------------------------------------------------------------
def download_dataset(destination: Path = DATA_DIR / DOWNLOAD_FILE) -> Path:
    """Download DBPEDIA_train.csv from the public Hugging Face mirror (no login needed)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    print(f"Downloading {DOWNLOAD_FILE} (~160 MB) from the public mirror:\n  {DOWNLOAD_URL}")
    try:
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=60) as response, partial.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        partial.unlink(missing_ok=True)
        hint = ""
        if "CERTIFICATE_VERIFY_FAILED" in str(error):
            hint = ("(macOS tip: if you installed Python from python.org, run 'Install Certificates.command' "
                    "in Applications > Python 3.x, then try again.)\n\n")
        raise DatasetError(f"Download failed: {error}\n{hint}\n" + manual_download_help()) from error
    partial.rename(destination)
    print(f"Saved data/{destination.name}")
    return destination


def find_dataset_files(data_path: Optional[Path], allow_download: bool) -> List[Path]:
    """CSV files to load: --data (file or folder) if given, otherwise the DBPEDIA files in data/."""
    if data_path is not None:
        if data_path.is_file():
            return [data_path]
        if data_path.is_dir():
            files = sorted(data_path.glob("*.csv"))
            if files:
                return files
        raise DatasetError(f"No CSV file found at {data_path}")
    files = [DATA_DIR / name for name in DATASET_FILES if (DATA_DIR / name).is_file()]
    if files:
        return files
    if allow_download:
        return [download_dataset()]
    raise DatasetError("The dataset is not in the data/ folder.\n\n" + manual_download_help())


def load_dataset(files: List[Path], required_columns: List[str]) -> pd.DataFrame:
    """Read and combine the CSV files (the files themselves are never modified)."""
    frames = []
    for path in files:
        try:
            columns = list(pd.read_csv(path, nrows=0).columns)  # header only
            missing = [c for c in required_columns if c not in columns]
            if missing:
                raise DatasetError(
                    f"{path.name} is missing the column(s) {missing}. Found columns: {columns}.\n"
                    "Use --text-column / --label-column / --filter-column to match your dataset."
                )
            frame = pd.read_csv(path, usecols=required_columns)  # only the columns we need (faster)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as error:
            raise DatasetError(f"Could not read {path.name} as CSV: {error}") from error
        frames.append(frame)
        print(f"  {path.name}: {len(frame):,} rows; columns in file: {columns}")
    return pd.concat(frames, ignore_index=True)


def inspect_dataset(frame: pd.DataFrame) -> Dict[str, int]:
    print(f"\nColumns: {list(frame.columns)}")
    missing = frame.isna().sum()
    print("Missing values per column: " + ", ".join(f"{c}={int(n)}" for c, n in missing.items()))
    return {c: int(n) for c, n in missing.items()}


# ---------------------------------------------------------------------------
# 3-5. Select athletes, map sports, clean text
# ---------------------------------------------------------------------------
def clean_text(text: object) -> str:
    """Light cleaning used for training AND for new biographies in the app:
    remove links and wiki-style brackets, keep letters, collapse spaces.
    (Lower-casing and stop-word removal happen inside the TF-IDF step.)"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)           # [1], [citation needed]
    text = re.sub(r"[^\w\s]|[\d_]", " ", text)          # punctuation, digits
    return re.sub(r"\s+", " ", text).strip()


def map_sport(label: str, use_mapping: bool) -> str:
    if not use_mapping:
        return label
    return SPORT_MAP.get(label, OTHER_LABEL)


def prepare_data(
    frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    filter_column: Optional[str],
    filter_value: Optional[str],
    use_mapping: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Return a table with columns 'text' and 'sport', plus a summary of each step."""
    summary: Dict[str, object] = {"rows_loaded": int(len(frame))}
    data = frame
    if filter_column:
        data = data[data[filter_column] == filter_value]
        summary["rows_after_filter"] = int(len(data))
        print(f"\nRows where {filter_column} == '{filter_value}': {len(data):,}")
        if data.empty:
            raise DatasetError(f"No rows have {filter_column} == '{filter_value}'.")

    before = len(data)
    data = data.dropna(subset=[text_column, label_column])
    summary["dropped_missing_text_or_label"] = int(before - len(data))

    original_labels = data[label_column].astype(str).str.strip()
    summary["original_label_counts"] = original_labels.value_counts().to_dict()
    if use_mapping:
        found = sorted(original_labels.unique())
        summary["label_mapping"] = {label: map_sport(label, True) for label in found}
        absent = [cls for cls in SPORT_MAP if cls not in found]
        if absent:
            print(f"Note: these classes are not in the data: {absent}")
            summary["mapped_classes_not_found"] = absent

    prepared = pd.DataFrame({
        "text": data[text_column].map(clean_text),
        "sport": original_labels.map(lambda label: map_sport(label, use_mapping)),
    })
    before = len(prepared)
    prepared = prepared[prepared["text"].str.split().str.len() >= MIN_WORDS]
    summary["dropped_too_short"] = int(before - len(prepared))
    before = len(prepared)
    prepared = prepared.drop_duplicates(subset="text")
    summary["dropped_duplicate_text"] = int(before - len(prepared))

    counts = prepared["sport"].value_counts()
    too_small = counts[counts < 2].index.tolist()
    if too_small:  # stratified splitting needs at least 2 examples per class
        prepared = prepared[~prepared["sport"].isin(too_small)]
        summary["dropped_classes_with_one_example"] = too_small
    if prepared["sport"].nunique() < 2:
        raise DatasetError("Need at least two sport categories to train a classifier.")
    summary["rows_used"] = int(len(prepared))
    summary["category_counts"] = prepared["sport"].value_counts().to_dict()
    return prepared.reset_index(drop=True), summary


# ---------------------------------------------------------------------------
# 6-8. Pipeline and training
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("tree", DecisionTreeClassifier(**TREE_PARAMS)),
    ])


def ordered_labels(labels) -> List[str]:
    present = list(dict.fromkeys(labels))
    return [c for c in CATEGORY_ORDER if c in present] + sorted(c for c in present if c not in CATEGORY_ORDER)


# ---------------------------------------------------------------------------
# 9. Evaluation
# ---------------------------------------------------------------------------
def evaluate(pipeline: Pipeline, x_test: pd.Series, y_test: pd.Series) -> Dict[str, object]:
    predicted = pipeline.predict(x_test)
    labels = ordered_labels(list(y_test) + list(predicted))
    macro = precision_recall_fscore_support(y_test, predicted, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_test, predicted, labels=labels, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "classification_report": classification_report(
            y_test, predicted, labels=labels, output_dict=True, zero_division=0),
        "classification_report_text": classification_report(
            y_test, predicted, labels=labels, digits=3, zero_division=0),
    }


def save_confusion_matrix(matrix: List[List[int]], labels: List[str], path: Path) -> None:
    values = np.array(matrix, dtype=float)
    share = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
    size = max(5.5, 1.1 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(values[i, j])}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.55 else "#1f1f1f")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted sport")
    ax.set_ylabel("Actual sport")
    ax.set_title("Decision Tree - confusion matrix (test set)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of the actual sport")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_text_report(results: Dict[str, object], path: Path) -> None:
    m = results["metrics"]
    d = results["data"]
    lines = [
        "Athlete Biography Classification - Decision Tree",
        "=" * 50,
        f"Trained at        : {results['trained_at']}",
        f"Dataset           : {results['dataset']['name']} ({results['dataset']['source']})",
        f"Files used        : {', '.join(results['dataset']['files'])}",
        f"Rows loaded       : {d['rows_loaded']:,}",
        f"Athlete rows      : {d.get('rows_after_filter', d['rows_loaded']):,}",
        f"Rows used         : {d['rows_used']:,} (after removing missing, short and duplicate texts)",
        f"Train / test rows : {results['split']['train_rows']:,} / {results['split']['test_rows']:,} "
        f"(stratified, random_state={RANDOM_STATE})",
        "",
        "Category counts (all rows used):",
        *[f"  {name:<14} {count:,}" for name, count in d["category_counts"].items()],
        "",
        "Test-set metrics:",
        f"  Accuracy             : {m['accuracy']:.4f}",
        f"  Precision (macro)    : {m['precision_macro']:.4f}   (weighted: {m['precision_weighted']:.4f})",
        f"  Recall (macro)       : {m['recall_macro']:.4f}   (weighted: {m['recall_weighted']:.4f})",
        f"  F1-score (macro)     : {m['f1_macro']:.4f}   (weighted: {m['f1_weighted']:.4f})",
        "",
        "Classification report:",
        m["classification_report_text"],
        "Confusion matrix (rows = actual, columns = predicted):",
        "  " + " | ".join(m["labels"]),
        *[f"  {label:<14} {row}" for label, row in zip(m["labels"], m["confusion_matrix"])],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Prediction helper (used by the dashboard)
# ---------------------------------------------------------------------------
def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load the trained pipeline. Only this project's own model file is ever loaded."""
    if not path.is_file():
        raise FileNotFoundError(f"Model not found at {path.relative_to(PROJECT_ROOT)}. Run: python train.py")
    return joblib.load(path)


def predict_sport(pipeline: Pipeline, biography: str) -> Tuple[str, Dict[str, float], int]:
    """Clean the text and predict. Returns (sport, {sport: probability}, known_word_count).

    Raises ValueError for empty or too-short input.
    """
    cleaned = clean_text(biography)
    if not cleaned:
        raise ValueError("Please enter a biography.")
    if len(cleaned.split()) < 3:
        raise ValueError("Please enter at least a few words (a sentence or more works best).")
    known_words = int(pipeline.named_steps["tfidf"].transform([cleaned]).nnz)
    sport = str(pipeline.predict([cleaned])[0])
    probabilities = pipeline.predict_proba([cleaned])[0]
    classes = [str(c) for c in pipeline.classes_]
    ranked = dict(sorted(zip(classes, map(float, probabilities)), key=lambda item: -item[1]))
    return sport, ranked, known_words


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the athlete biography Decision Tree classifier.")
    parser.add_argument("--data", type=Path, help="CSV file or folder of CSV files (default: DBPEDIA files in data/)")
    parser.add_argument("--text-column", default=TEXT_COLUMN)
    parser.add_argument("--label-column", default=LABEL_COLUMN)
    parser.add_argument("--filter-column", default=FILTER_COLUMN, help="Use '' to keep all rows")
    parser.add_argument("--filter-value", default=FILTER_VALUE)
    parser.add_argument("--no-mapping", action="store_true", help="Use the label column as-is (no sport grouping)")
    parser.add_argument("--sample", type=int, help="Train on a stratified sample of N rows (quick test)")
    parser.add_argument("--no-download", action="store_true", help="Never try to download the dataset")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    filter_column = args.filter_column or None
    required = [args.text_column, args.label_column] + ([filter_column] if filter_column else [])
    try:
        files = find_dataset_files(args.data, allow_download=not args.no_download)
        print("Loading dataset:")
        frame = load_dataset(files, required)
        missing_values = inspect_dataset(frame)
        data, summary = prepare_data(frame, args.text_column, args.label_column, filter_column,
                                     args.filter_value, use_mapping=not args.no_mapping)
    except DatasetError as error:
        print(f"\nDATASET PROBLEM\n{error}", file=sys.stderr)
        return 1
    summary["missing_values"] = missing_values

    if args.sample and args.sample < len(data):
        data, _ = train_test_split(data, train_size=args.sample, stratify=data["sport"], random_state=RANDOM_STATE)
        summary["sampled_rows"] = int(len(data))
        print(f"Using a stratified sample of {len(data):,} rows (--sample)")

    print("\nRows per category:")
    for name, count in data["sport"].value_counts().items():
        print(f"  {name:<14} {count:,}")

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["sport"], test_size=TEST_SIZE, stratify=data["sport"], random_state=RANDOM_STATE)
    print(f"\nTraining on {len(x_train):,} biographies, testing on {len(x_test):,} ...")
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    tree = pipeline.named_steps["tree"]
    print(f"Tree depth: {tree.get_depth()}, leaves: {tree.get_n_leaves()}, "
          f"vocabulary: {len(pipeline.named_steps['tfidf'].vocabulary_):,} words")

    metrics = evaluate(pipeline, x_test, y_test)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_path)
    results = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": DATASET_NAME,
            "source": DATASET_SOURCE,
            "mirror": DATASET_MIRROR,
            "files": [p.name for p in files],
            "text_column": args.text_column,
            "label_column": args.label_column,
            "filter": f"{filter_column} == {args.filter_value}" if filter_column else None,
        },
        "data": summary,
        "split": {"train_rows": int(len(x_train)), "test_rows": int(len(x_test)),
                  "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": True},
        "model": {"tfidf": {k: v for k, v in TFIDF_PARAMS.items()}, "tree": TREE_PARAMS,
                  "tree_depth": int(tree.get_depth()), "tree_leaves": int(tree.get_n_leaves()),
                  "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
                  "scikit_learn_version": sklearn.__version__},
        "metrics": metrics,
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / METRICS_JSON.name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_text_report(results, args.reports_dir / METRICS_TXT.name)
    save_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], args.reports_dir / CONFUSION_PNG.name)

    print("\n" + "=" * 60)
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Precision (macro)  : {metrics['precision_macro']:.4f}")
    print(f"Recall (macro)     : {metrics['recall_macro']:.4f}")
    print(f"F1-score (macro)   : {metrics['f1_macro']:.4f}")
    print("=" * 60)
    print(metrics["classification_report_text"])
    print(f"Saved model   : {args.model_path}")
    print(f"Saved reports : {args.reports_dir / METRICS_TXT.name}, {CONFUSION_PNG.name}, {METRICS_JSON.name}")
    print("Next          : streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
