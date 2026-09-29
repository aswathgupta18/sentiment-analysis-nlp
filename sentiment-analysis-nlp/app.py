import json

import pandas as pd
import streamlit as st

from src.config import COMPARISON_PATH, CONFUSION_PATH, METRICS_PATH, MODEL_PATH
from src.predict import predict_sentiment

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="centered")

EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}

st.title("💬 Review Sentiment Analyzer")
st.caption("TF-IDF + a scikit-learn classifier. Type a review and get the sentiment in real time.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

tab_predict, tab_model = st.tabs(["Predict", "About the model"])

with tab_predict:
    examples = {
        "": "",
        "Positive": "The camera quality is fantastic and the battery lasts all day.",
        "Negative": "The product arrived damaged and support was no help.",
        "Neutral": "The package arrived on Monday in a black box.",
        "Negation": "This is not good at all, I do not recommend it.",
    }
    choice = st.selectbox("Try an example (optional)", list(examples))
    text = st.text_area("Your review", value=examples[choice], height=130,
                        placeholder="Type or paste a review here...")

    if st.button("Analyze", type="primary"):
        if len(text.strip()) < 3:
            st.warning("Please enter a longer review.")
        else:
            label, conf, scores = predict_sentiment(text)
            st.subheader(f"{EMOJI[label]} {label}")
            st.metric("Confidence", f"{conf:.0%}")
            st.bar_chart(pd.Series(scores, name="probability"))
            if conf < 0.6:
                st.info("The model is not very sure about this one. Mixed or sarcastic "
                        "reviews are hard for a TF-IDF model.")

with tab_model:
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        c1, c2, c3 = st.columns(3)
        c1.metric("Model", m["best_model"])
        c2.metric("Test accuracy", f"{m['test_metrics']['accuracy']:.1%}")
        c3.metric("Macro F1", f"{m['test_metrics']['f1_macro']:.1%}")
        st.write(f"Trained on **{m['train_size']}** reviews, tested on **{m['test_size']}** "
                 f"({m['split']}). Inference: ~{m['inference_ms_per_review']} ms per review.")
    if COMPARISON_PATH.exists():
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(COMPARISON_PATH), hide_index=True)
    if CONFUSION_PATH.exists():
        st.subheader("Confusion matrix")
        st.image(str(CONFUSION_PATH))
