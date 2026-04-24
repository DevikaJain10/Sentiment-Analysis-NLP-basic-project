#  Sentiment Analysis of IMDb Movie Reviews

##  Project Overview
This project builds a Natural Language Processing (NLP) pipeline to classify movie reviews as **positive** or **negative** using the IMDb dataset from kaggle.  

The focus is on understanding the **complete NLP workflow**, including preprocessing, POS tagging, feature extraction, and model comparison.

---

##  Key Features
- Text preprocessing (cleaning, tokenization, stopword removal)
- POS (Part-of-Speech) tagging using NLTK
- TF-IDF feature extraction with unigrams and bigrams
- Comparison of multiple models:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
- Handling **negation words** (e.g., "not good") to improve accuracy
- Cross-validation for robust evaluation
- Custom function to predict sentiment on unseen input

---
Tech stack: 
Python, NLTK ,Scikit-learn ,Pandas, NumPy
##  NLP Pipeline

Raw Text
   ↓
   Preprocessing
      ├── Lowercasing
      ├── HTML Removal
      └── Punctuation Removal
   ↓
   Tokenization
   ↓
   Stopword Removal
      └── Negation Preserved (e.g., "not")
   ↓
   POS Tagging
   ↓
   TF-IDF Feature Extraction
      └── Unigrams + Bigrams
   ↓
   Model Training
      ├── Naive Bayes
      ├── Logistic Regression
      └── Support Vector Machine (SVM)
   ↓
   Evaluation & Prediction

## Future Improvements

Use deep learning models (LSTM, BERT)

Improve sarcasm detection

Deploy as a web app (Streamlit/Flask)
