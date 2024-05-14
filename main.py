import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Load the previously saved model
model = load_model('pneumonia_detector.h5')

def load_and_prepare_image(uploaded_file):
    # Load the uploaded image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Normalize the image
    return img_array

def predict_image(img_array):
    prediction = model.predict(img_array)
    return prediction[0][0]

# Set up the title of the app
st.title('Pneumonia Detection from Chest X-Rays')

# Instructions
st.write("This tool can help you detect pneumonia from chest X-ray images. Please upload an X-ray image.")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
     # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded X-ray Image', use_column_width=True)
    img_array = load_and_prepare_image(uploaded_file)
    
    # On clicking the Predict button, make the prediction and display it
    if st.button('Predict'):
        result = predict_image(img_array)
        if result > 0.5:
            st.write('Prediction: **Pneumonia**')
            st.write(f'Probability of Pneumonia: {result:.2f}')
        else:
            st.write('Prediction: **Normal**')
            st.write(f'Probability of Normal: {1 - result:.2f}')

# streamlit run main.py --server.enableXsrfProtection false 
# command to run streamlit app