import streamlit as st
import joblib
import pandas as pd
from data_preprocessing import preprocess_text

# Load models safely
@st.cache_resource
def load_models():
    model = joblib.load('models/emotion_classifier.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

def main():
    st.set_page_config(page_title="Emotion Detector", page_icon="🧠")
    
    st.title("AI-Powered Emotion Detection 🧠")
    st.markdown("Type a sentence below to see what emotion the AI detects!")
    
    # Load model
    try:
        model, vectorizer = load_models()
    except Exception as e:
        st.error("Error loading model. Please ensure the models directory contains the .pkl files.")
        return

    label_mapping = {0: 'Sadness 😢', 1: 'Joy 😄', 2: 'Love 🥰', 3: 'Anger 😡', 4: 'Fear 😨', 5: 'Surprise 😲'}
    
    # User Input
    user_input = st.text_area("Enter your text:", "I am completely amazed and shocked by this news!")
    
    if st.button("Predict Emotion"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                cleaned_text = preprocess_text(user_input)
                features = vectorizer.transform([cleaned_text])
                
                prediction = model.predict(features)[0]
                predicted_emotion = label_mapping.get(prediction, "Unknown")
                
                probabilities = model.predict_proba(features)[0]
                
                st.success(f"**Predicted Emotion:** {predicted_emotion}")
                
                # Show probabilities chart
                st.subheader("Confidence Levels")
                prob_df = pd.DataFrame({
                    "Emotion": [label_mapping[i] for i in range(6)],
                    "Probability": probabilities * 100
                })
                prob_df = prob_df.sort_values(by="Probability", ascending=True)
                
                st.bar_chart(prob_df.set_index("Emotion"))

if __name__ == '__main__':
    main()
