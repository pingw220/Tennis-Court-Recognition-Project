import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = load_model('my_model.h5')

# Define a function to preprocess the images
def preprocess_image(img):
    img = img.resize((150, 150))  # Example resize to match model's expected input
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Rescale pixel values
    return img

# Create a file uploader to upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    display_image = image.load_img(uploaded_file)
    st.image(display_image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(display_image)
    
    # Predict the class of the uploaded image
    prediction = model.predict(preprocessed_image)
    predicted_class_indices = np.argmax(prediction, axis=1)  # Example for categorical classification
    
    class_names = ['clay', 'grass', 'hard']  # Example: class names
    
    # Translate indices to names
    predicted_class_names = [class_names[i] for i in predicted_class_indices]
    print(f"Predicted court type: {predicted_class_names[0]}")

    # Display the prediction
    st.write(f"Predicted court type: {predicted_class_names[0]}")