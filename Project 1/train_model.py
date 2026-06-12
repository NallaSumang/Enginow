import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from data_preprocessing import preprocess_text

def main():
    print("Loading dair-ai/emotion dataset from Hugging Face...")
    # Load dataset
    dataset = load_dataset("dair-ai/emotion", "split")
    
    # Convert to pandas dataframes
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    # Map label integers to string names for clarity
    label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    train_df['label_name'] = train_df['label'].map(label_mapping)
    test_df['label_name'] = test_df['label'].map(label_mapping)
    
    print("Preprocessing text data...")
    # Apply preprocessing
    train_df['clean_text'] = train_df['text'].apply(preprocess_text)
    test_df['clean_text'] = test_df['text'].apply(preprocess_text)
    
    print("Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    y_train = train_df['label']
    
    X_test = vectorizer.transform(test_df['clean_text'])
    y_test = test_df['label']
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=[label_mapping[i] for i in range(6)]))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[label_mapping[i] for i in range(6)], yticklabels=[label_mapping[i] for i in range(6)])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Emotion Detection')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix plot to confusion_matrix.png")
    
    print("Saving model and vectorizer...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/emotion_classifier.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("Model and vectorizer saved in 'models/' directory.")

if __name__ == "__main__":
    main()
