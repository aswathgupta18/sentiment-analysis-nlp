"""Text cleaning as a scikit-learn transformer, so it travels inside the saved Pipeline."""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


def _ensure_nltk():
    for pkg, path in [("stopwords", "corpora/stopwords"),
                      ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


# Negations carry sentiment ("not good" is not "good"), so we keep them.
NEGATIONS = {"not", "no", "nor", "never", "cannot", "without", "hardly"}
NEGATED_STEMS = {"don", "didn", "doesn", "isn", "wasn", "weren", "won", "wouldn",
                 "couldn", "shouldn", "aren", "haven", "hasn", "hadn"}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HTML_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")

CONTRACTIONS = [
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "cannot"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'ve\b", " have"),
    (r"'ll\b", " will"), (r"'m\b", " am"), (r"'d\b", " would"),
]


class TextCleaner(BaseEstimator, TransformerMixin):
    """lowercase, strip noise, expand contractions, tokenize, drop stop words, lemmatize."""

    def __init__(self, keep_negations=True, lemmatize=True):
        self.keep_negations = keep_negations
        self.lemmatize = lemmatize

    def _setup(self):
        _ensure_nltk()
        stops = set(stopwords.words("english"))
        if self.keep_negations:
            stops -= NEGATIONS
            stops -= NEGATED_STEMS
        self._stops = stops
        self._lemmatizer = WordNetLemmatizer()

    def clean_one(self, text):
        if not hasattr(self, "_stops"):
            self._setup()
        text = str(text).lower()
        text = URL_RE.sub(" ", text)
        text = MENTION_RE.sub(" ", text)
        text = HTML_RE.sub(" ", text)
        for pattern, repl in CONTRACTIONS:
            text = re.sub(pattern, repl, text)
        text = NON_ALPHA_RE.sub(" ", text)
        tokens = [t for t in text.split() if t not in self._stops and len(t) > 1]
        if self.lemmatize:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_one(t) for t in X]

    def __getstate__(self):
        # NLTK objects are not saved with the model; they are rebuilt on first use
        state = self.__dict__.copy()
        state.pop("_stops", None)
        state.pop("_lemmatizer", None)
        return state
