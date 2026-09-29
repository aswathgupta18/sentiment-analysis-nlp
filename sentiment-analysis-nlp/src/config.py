from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sample_reviews.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
REPORT_PATH = MODEL_DIR / "classification_report.txt"
CONFUSION_PATH = ROOT / "assets" / "confusion_matrix.png"

TEXT_COL = "text"
LABEL_COL = "label"
LABELS = ["Negative", "Neutral", "Positive"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
