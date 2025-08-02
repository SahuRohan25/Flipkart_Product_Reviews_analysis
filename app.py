# Streamlit Sentiment Analysis App for Flipkart Product Reviews

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# Load pre-trained model and vectorizer
def load_model():
    model = joblib.load("logistic_models.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

# Main Streamlit app
def main():
    st.title("🛍️ Flipkart Product Review Sentiment Analyzer")
    st.write("Enter a product review to predict its sentiment:")

    user_input = st.text_area("Your Review", "Great product! Works as expected.")

    if st.button("Analyze Sentiment"):
        model, vectorizer = load_model()
        clean_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([clean_input])
        prediction = model.predict(vectorized_input)[0]

        st.subheader("🔍 Sentiment Prediction:")
        if prediction == 'positive':
            st.success("😊 Positive Review")
        elif prediction == 'neutral':
            st.info("😐 Neutral Review")
        else:
            st.error("😠 Negative Review")

if __name__ == "__main__":
    main()
