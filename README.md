# AI & Machine Learning Projects Portfolio

Welcome to my AI & Machine Learning portfolio!

This repository serves as a centralized collection of my machine learning, deep learning, and natural language processing projects. Each project is organized into its own dedicated folder with its own source code, datasets, and documentation.

---

## 📂 Projects Overview

### [Project 1: AI-Powered Emotion Detection from Text](./Project%201)
**Status:** ✅ Completed  
**Tech Stack:** Python, Scikit-Learn, NLTK, Pandas, Hugging Face `datasets`

**Description:**  
An end-to-end Natural Language Processing (NLP) pipeline that analyzes text to predict the underlying emotion (Sadness, Joy, Love, Anger, Fear, or Surprise). The project demonstrates proficiency in text preprocessing (tokenization, stopword removal, lemmatization), TF-IDF feature extraction, and training a Logistic Regression model. It includes both interactive Jupyter Notebooks for EDA/presentation and production-ready Python scripts for automated training and command-line inference.

---

### [Project 2: Image Classification Using Convolutional Neural Networks (CNNs)](./Project%202)
**Status:** ✅ Completed  
**Tech Stack:** Python, TensorFlow, Keras, NumPy

**Description:**  
A lightweight image classification pipeline using a Convolutional Neural Network (CNN) on the MNIST dataset. The project demonstrates the construction, training, and evaluation of a minimal sequential model optimized for speed and low memory usage, achieving high accuracy in very few epochs.

---

### [Project 3: AI-Powered Resume Ranking System](./Project%203)
**Status:** ✅ Completed  
**Tech Stack:** Python, Streamlit, SpaCy, Scikit-Learn, PyPDF2, Pandas

**Description:**  
An interactive NLP application built with Streamlit that automatically screens and ranks candidate resumes against a target job description. It extracts text from PDF files, performs text preprocessing with SpaCy, and computes the Cosine Similarity between resumes and the job description using TF-IDF vectorization to identify the best fit.

---

## 🚀 How to Navigate
- Click on any project folder above to dive into its specific implementation.
- Each project folder contains the fully executable scripts, models, and notebooks required to run the code locally.

## 🛠️ General Setup
If you want to clone this repository and run the projects locally, you will generally need Python installed along with the following libraries:
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter joblib datasets tensorflow streamlit spacy PyPDF2
python -m spacy download en_core_web_sm
```

---
*More projects will be added to this repository over time.*
