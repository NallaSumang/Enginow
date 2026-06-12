import joblib
import sys
from data_preprocessing import preprocess_text

def predict_emotion(text, model_path='models/emotion_classifier.pkl', vectorizer_path='models/tfidf_vectorizer.pkl'):
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError:
        print("Error: Model or vectorizer not found. Please run train_model.py first.")
        return

    label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Extract features
    features = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(features)[0]
    predicted_emotion = label_mapping.get(prediction, "Unknown")
    
    # Probabilities
    probabilities = model.predict_proba(features)[0]
    prob_dict = {label_mapping[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}
    
    print(f"\nInput Text: '{text}'")
    print(f"Predicted Emotion: **{predicted_emotion.upper()}**")
    print("Probabilities:")
    for emotion, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {emotion.capitalize()}: {prob}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text_input = " ".join(sys.argv[1:])
    else:
        text_input = "I am so happy and excited about this new project!"
    
    predict_emotion(text_input)
