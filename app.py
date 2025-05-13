import streamlit as st
import re, nltk, joblib
from nltk.tokenize import wordpunct_tokenize
import os, certifi

# Optional: Fix SSL issues on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load model & vectorizer
model      = joblib.load('./train/sentiment_model.pkl')
vectorizer = joblib.load('./train/vectorizer.pkl')

# Optional: simple rules-based lemmatizer
def custom_lemmatizer(word):
    if word.endswith("ing") or word.endswith("ed"):
        return word[:-3]
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word

def preprocess(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = wordpunct_tokenize(text)  # ✅ No punkt required
        tokens = [custom_lemmatizer(token) for token in tokens if token.isalpha()]
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        raise e

# Streamlit UI
st.title('Simple Sentiment Analysis')
input_text = st.text_area("Masukkan Twit:")

if st.button('Prediksi') and input_text.strip():
    clean_text       = preprocess(input_text)
    vectorized_text  = vectorizer.transform([clean_text])
    prediction       = model.predict(vectorized_text)[0]
    st.write(f"Sentimen Twit: **{prediction.capitalize()}**")
