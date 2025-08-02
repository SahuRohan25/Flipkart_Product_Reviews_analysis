# 🛍️ Flipkart Product Review Sentiment Analyzer

This is a Streamlit-based web application that uses Machine Learning to analyze the **sentiment** (Positive, Neutral, or Negative) of product reviews on Flipkart.

## 📌 Features

- Accepts user input for any Flipkart-style product review.
- Preprocesses and cleans the text.
- Uses a **Logistic Regression model** trained on TF-IDF features.
- Outputs predicted sentiment with emojis for better understanding.

## 🔧 Tech Stack

- Python
- scikit-learn
- Pandas
- NumPy
- Joblib

## 📂 Project Structure

- Streamlit📁 flipkart_product_review_app/
│
├── app.py # Main Streamlit app
├── logistic_model.pkl # Trained Logistic Regression model
├── tfidf_vectorizer.pkl # Trained TF-IDF vectorizer
├── requirements.txt # Required libraries
└── README.md # Project documentation

