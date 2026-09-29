import joblib

from .config import MODEL_PATH

_model = None


def load_model(path=MODEL_PATH):
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def predict_sentiment(text):
    """Return (label, confidence, {label: probability}) for one review."""
    model = load_model()
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    scores = {str(c): float(p) for c, p in zip(classes, probs)}
    label = max(scores, key=scores.get)
    return label, scores[label], scores


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "The camera quality is fantastic."
    label, conf, scores = predict_sentiment(text)
    print(f"{label} ({conf:.0%})", scores)
