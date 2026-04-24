#  Sentiment Analysis of IMDb Movie Reviews

##  Project Overview
This project builds a Natural Language Processing (NLP) pipeline to classify movie reviews as **positive** or **negative** using the IMDb dataset from kaggle.  

The focus is on understanding the **complete NLP workflow**, including preprocessing, POS tagging, feature extraction, and model comparison.

---
## Dataset Setup
to download the dataset: use this link (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
OR
You can download the IMDb dataset using the following code:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

print("Path to dataset files:", path)
```

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

Raw Text:
   ->Preprocessing
      -Lowercasing
      - HTML Removal
      - Punctuation Removal
   
   ->Tokenization
   
   ->Stopword Removal
      -Negation Preserved (e.g., "not")
   
   ->POS Tagging
   
   ->TF-IDF Feature Extraction
      - Unigrams + Bigrams
   
   ->Model Training
      - Naive Bayes
      - Logistic Regression
      - Support Vector Machine (SVM)
   
   ->Evaluation & Prediction

## Future Improvements

1)Use deep learning models (LSTM, BERT)

2)Improve sarcasm detection

3)Deploy as a web app (Streamlit/Flask)
