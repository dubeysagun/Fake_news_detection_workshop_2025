import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To save and load the model
import re
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Paths to save/load models
TEXT_MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# Load and preprocess text data
def load_and_preprocess_text_data():
    data = pd.read_csv('news2.csv', index_col=0)
    
    # Drop unnecessary columns
    data = data.drop(["title", "subject", "date"], axis=1)
    
    # Handle missing values
    data = data.dropna()
    
    # Shuffle dataset
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Text preprocessing
    def preprocess_text(text_data):
        preprocessed_text = []
        for sentence in tqdm(text_data):
            sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
            preprocessed_text.append(' '.join(token.lower()
                                              for token in str(sentence).split()
                                              if token not in stopwords.words('english')))
        return preprocessed_text
    
    data['text'] = preprocess_text(data['text'].values)
    
    return data

# Train or load the text classifier
def train_text_classifier(train_data):
    try:
        model = joblib.load(TEXT_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        st.write("Loaded pre-trained text classifier.")
    except FileNotFoundError:
        st.write("Training new text classifier...")
        
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(train_data['text'])
        y = train_data['class']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Train DecisionTreeClassifier (matching the .ipynb file)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        
        # Model accuracy
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        st.write(f"Training Accuracy: {train_acc * 100:.2f}%")
        st.write(f"Validation Accuracy: {val_acc * 100:.2f}%")
        
        # Save the trained model and vectorizer
        joblib.dump(model, TEXT_MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        st.write("Text classifier model saved.")
    
    return model, vectorizer

# Predict text function
def predict_text(text, model, vectorizer):
    text_vectorized = vectorizer.transform([text.lower()])
    prediction = model.predict(text_vectorized)
    return "Real" if prediction == 1 else "Fake"

# Main Streamlit App
def main():
    st.title("Fake News Detection System")
    st.write("Enter news text to predict if it's fake or real")
    
    # Load and train models (if not already saved)
    train_data = load_and_preprocess_text_data()
    text_model, vectorizer = train_text_classifier(train_data)
    
    # User input (text only)
    user_input = st.text_area("Enter the news text")
    if st.button("Predict"):
        result = predict_text(user_input, text_model, vectorizer)
        st.write(f"The news is: {result}")

if __name__ == '__main__':
    main()
