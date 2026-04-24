import streamlit as st
import re
import pickle
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Page config
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

.title {
    text-align: center;
    color: #facc15;
    font-size: 42px;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 25px;
}

.positive {
    background-color: #dcfce7;
    color: #166534;
}

.negative {
    background-color: #fee2e2;
    color: #991b1b;
}

.info-box {
    background-color: #1e293b;
    color: #e2e8f0;
    padding: 18px;
    border-radius: 12px;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"no", "not", "nor", "never"}

def handle_negation(text):
    text = text.replace("not good", "not_good")
    text = text.replace("not bad", "not_bad")
    text = text.replace("not great", "not_great")
    text = text.replace("not amazing", "not_amazing")
    text = text.replace("not worth", "not_worth")
    text = text.replace("never good", "never_good")
    return text

def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = handle_negation(text)
    text = re.sub(r"[^a-zA-Z_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

# Header
st.markdown('<div class="title">🎬 IMDb Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Analyze whether a movie review is positive or negative using NLP + SVM</div>',
    unsafe_allow_html=True
)

# Input card
st.markdown("### ✍️ Enter a Movie Review")

user_input = st.text_area(
    "",
    placeholder="Example: This movie was not good. The story was boring...",
    height=160
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

if predict_button:
    if user_input.strip() == "":
        st.warning("Please enter a movie review first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned_text = preprocess(user_input)
            vectorized_text = tfidf.transform([cleaned_text])
            prediction = svm_model.predict(vectorized_text)[0]

        if prediction == "positive":
            st.markdown(
                '<div class="result-box positive">😊 Positive Review</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box negative">😞 Negative Review</div>',
                unsafe_allow_html=True
            )

        with st.expander("See processed text"):
            st.write(cleaned_text)

# Example section
st.markdown("---")
st.markdown("### 🧪 Try Example Reviews")

example_col1, example_col2 = st.columns(2)

with example_col1:
    st.info("This movie was fantastic! The acting and story were amazing.")

with example_col2:
    st.info("This movie was not good. The plot was boring and slow.")

# About section
st.markdown("""
<div class="info-box">
<h3>📌 About this Project</h3>
<p>This app uses an NLP pipeline with preprocessing, negation handling, TF-IDF feature extraction, and an SVM classifier to predict movie review sentiment.</p>
<p><b>Tech Stack:</b> Python, NLTK, Scikit-learn, Streamlit</p>
</div>
""", unsafe_allow_html=True)
