import pandas as pd

from .config import DATA_PATH, LABEL_COL, LABELS, TEXT_COL

# Lets you drop in other datasets (Kaggle airline tweets, Amazon reviews, etc.)
_LABEL_ALIASES = {
    "negative": "Negative", "neg": "Negative", "0": "Negative",
    "neutral": "Neutral", "neu": "Neutral", "1": "Neutral",
    "positive": "Positive", "pos": "Positive", "2": "Positive",
}


def load_dataset(path=DATA_PATH, text_col=None, label_col=None):
    """Load a CSV, keep text + label, normalise labels, drop empties and duplicates."""
    df = pd.read_csv(path)
    text_col = text_col or (TEXT_COL if TEXT_COL in df else None)
    label_col = label_col or (LABEL_COL if LABEL_COL in df else None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"CSV needs '{TEXT_COL}' and '{LABEL_COL}' columns (or pass --text-col / --label-col). "
            f"Found: {list(df.columns)}")

    df = df[[text_col, label_col]].rename(columns={text_col: TEXT_COL, label_col: LABEL_COL})
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower().map(_LABEL_ALIASES)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[TEXT_COL].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
    assert set(df[LABEL_COL]) <= set(LABELS), "unexpected labels after mapping"
    return df


def describe(df):
    counts = df[LABEL_COL].value_counts().reindex(LABELS).fillna(0)
    return {
        "rows": int(len(df)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_share_pct": {k: round(100 * v / len(df), 1) for k, v in counts.items()},
    }
