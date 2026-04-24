#   Sentiment Analysis of IMDb Movie Reviews

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://sentiment-analysis-nlp-basic-project-kt3srvjmc6uqrzw9gejj3x.streamlit.app/)

##  Live Demo
https://sentiment-analysis-nlp-basic-project-kt3srvjmc6uqrzw9gejj3x.streamlit.app/

---

##  Project Overview

This project builds an end-to-end NLP pipeline to classify IMDb movie reviews as **positive** or **negative**.

It includes text preprocessing, POS tagging, TF-IDF feature extraction, and sentiment classification using an SVM model.  
The project is deployed as an interactive web application using Streamlit.

---

##  Features

- End-to-end NLP pipeline
- POS tagging using NLTK
- TF-IDF feature extraction (unigrams + bigrams)
- Negation handling for improved accuracy
- Model comparison (Naive Bayes, Logistic Regression, SVM)
- Cross-validation for model robustness
- Interactive Streamlit app for real-time predictions

---

##  NLP Pipeline

```text
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
````

---

##  Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Naive Bayes         | ~78–82%  |
| Logistic Regression | ~84–86%  |
| SVM (Best)          | ~88–89%  |

> SVM performed best due to its effectiveness in handling high-dimensional sparse text data.

---

## Key Insights

* Negation words like **"not"** are crucial for sentiment understanding
* Bigrams help capture phrases like:

  * *not good*
  * *very bad*
* Positive reviews contain more **adjectives (JJ)** such as *amazing, great*
* Model errors often occur in:

  * Sarcasm
  * Mixed sentiment sentences

---

##  Example Predictions

```python
predict_sentiment("This movie was fantastic!") → positive
predict_sentiment("This movie was not good.") → negative
predict_sentiment("The acting was great but the story was boring.") → mixed/varies
```

---

##  Dataset

IMDb Movie Review Dataset (50K reviews)
Used a subset (~5000 reviews) for efficient training and experimentation.

---

##  How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/DevikaJain10/Sentiment-Analysis-NLP-basic-project.git
cd Sentiment-Analysis-NLP-basic-project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Open the link shown in the terminal (usually [http://localhost:8501](http://localhost:8501))

---

##  Tech Stack

* Python
* NLTK
* Scikit-learn
* Pandas, NumPy
* Streamlit

---

## Future Improvements

* Use deep learning models (LSTM, BERT)
* Improve sarcasm detection
* Add confidence score for predictions

---

## Author

Devika Jain


