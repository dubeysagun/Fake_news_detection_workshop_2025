import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load stopwords only once
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

# Paths to saved model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Load pre-trained model & vectorizer
@st.cache_resource()
def load_model_and_vectorizer():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or Vectorizer not found. Train and save them first.")
        return None, None

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = " ".join(word.lower() for word in text.split() if word not in stop_words)
    return text

# Prediction function
def predict_text(text, model, vectorizer):
    if not model or not vectorizer:
        return "Model not available"
    text_vectorized = vectorizer.transform([preprocess_text(text)])
    prediction = model.predict(text_vectorized)
    return "Real" if prediction == 1 else "Fake"

# Streamlit UI
def main():
    st.title("🚀 Fake News Detection System")
    st.write("🔍 Enter a news article to check if it's Fake or Real.")

    # Load pre-trained model & vectorizer
    model, vectorizer = load_model_and_vectorizer()

    # User input
    user_input = st.text_area("✍️ Enter the news text")
    if st.button("📊 Predict"):
        result = predict_text(user_input, model, vectorizer)
        st.write(f"📰 The news is: **{result}**")

if __name__ == "__main__":
    main()
