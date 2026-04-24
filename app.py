import streamlit as st
import re
import pickle
from nltk.tokenize import word_tokenize

# Load saved model and vectorizer
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Stopwords (same as training)
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"no", "not", "nor", "never"}

# Preprocessing (same as before)
def handle_negation(text):
    text = text.replace("not good", "not_good")
    text = text.replace("not bad", "not_bad")
    return text

def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = handle_negation(text)
    text = re.sub(r"[^a-zA-Z_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]

    return " ".join(tokens)

# UI
st.title(" IMDb Sentiment Analyzer")

user_input = st.text_area("Enter a movie review:")

if st.button("Predict Sentiment"):
    cleaned = preprocess(user_input)
    vec = tfidf.transform([cleaned])
    prediction = svm_model.predict(vec)[0]

    st.write("### Prediction:", prediction)
