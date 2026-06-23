import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# Set page config
st.set_page_config(page_title="Digit Classifier", page_icon="🔢")

import train_model

# Load model safely
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'mnist_cnn.keras')
    if not os.path.exists(model_path):
        st.warning("Model not found! Training a lightweight model now (this takes ~1 minute). Please wait...")
        train_model.main()
    model = tf.keras.models.load_model(model_path)
    return model

def main():
    st.title("AI-Powered Digit Classifier 🔢")
    st.markdown("Upload an image of a handwritten digit (0-9) and the CNN will predict it!")
    
    # Load model
    model = load_model()

    # Image Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=200)
        
        if st.button("Predict Digit"):
            with st.spinner("Analyzing..."):
                # Preprocess the image
                # 1. Convert to grayscale
                img_gray = ImageOps.grayscale(image)
                # 2. Resize to 28x28 (MNIST expected size)
                img_resized = img_gray.resize((28, 28))
                # 3. Invert colors (MNIST is white on black background, usually uploaded images are black on white)
                # Check if image is mostly white, if so invert. Let's just assume we might need to invert for typical drawing.
                # A robust way is to just do inverted grayscale if the background is light.
                img_array = np.array(img_resized)
                if np.mean(img_array) > 127: # Light background
                    img_array = 255 - img_array
                
                # 4. Normalize to [0, 1]
                img_normalized = img_array / 255.0
                
                # 5. Reshape for the model: (1, 28, 28, 1)
                img_input = img_normalized.reshape(1, 28, 28, 1)
                
                # Predict
                predictions = model.predict(img_input)
                predicted_digit = np.argmax(predictions)
                confidence = np.max(predictions) * 100
                
                st.success(f"**Predicted Digit:** {predicted_digit}")
                st.info(f"**Confidence:** {confidence:.2f}%")
                
                # Display probability chart
                st.subheader("Prediction Probabilities")
                st.bar_chart(predictions[0])

if __name__ == '__main__':
    main()
