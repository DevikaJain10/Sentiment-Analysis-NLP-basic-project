
![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-brightgreen)
#  Sentiment Analysis of IMDb Movie Reviews
## 🌐 Live Demo
https://sentiment-analysis-nlp-basic-project-kt3srvjmc6uqrzw9gejj3x.streamlit.app/

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

##  How to Run

1. Clone the repository:
```
git clone <your-repo-link>
cd <repo-name>
```
2. Install dependencies:
```
   pip install -r requirements.txt
```
3. Run the script:
   ```
   python nlp_basic_project.py
   ```

## 📌 Project Overview

This project builds an end-to-end NLP pipeline to classify IMDb movie reviews as **positive** or **negative**.

It includes preprocessing, POS tagging, TF-IDF feature extraction, and sentiment classification using an SVM model.  
The project is deployed as an interactive web app using Streamlit.

##  Sample Output
Input: "This movie was fantastic!"
Output: positive

Input: "This movie was not good."
Output: negative

##  Key Features
- Interactive Streamlit web app for real-time predictions
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
