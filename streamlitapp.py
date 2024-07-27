import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the disease detection model
leaf_segmentation_model = load_model('my_model.keras')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

def detect_disease(image):
    # Blur the input image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    
    # Define the color threshold
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([10, 255, 255])
    
    # Threshold the image
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    thresholded_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
    
    # Apply background subtraction
    background_model = cv2.createBackgroundSubtractorMOG2()
    foreground_mask = background_model.apply(thresholded_image)
    foreground_image = cv2.bitwise_and(thresholded_image, thresholded_image, mask=foreground_mask)
    
    # Perform leaf segmentation prediction
    prediction = leaf_segmentation_model.predict(foreground_image.reshape(1, 100, 100, 3))
    
    # Decode the predicted label
    predicted_class_index = np.argmax(prediction)
    predicted_disease_label = label_encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_disease_label

def preprocess_image(image):
    # Resize the image to match the input shape of your model
    image = cv2.resize(image, (100, 100))
    # Convert the image to numpy array
    image = np.array(image)
    # Perform any additional preprocessing steps if required by your model
    return image

def main():
    st.title("Tulsi Leaf Disease Detection")

    uploaded_file = st.file_uploader("Upload an image of a Tulsi leaf", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert the PIL image to numpy array
        image_np = np.array(image)

        # Preprocess the image
        preprocessed_image = preprocess_image(image_np)

        # Detect disease
        disease_label = detect_disease(preprocessed_image)

        st.success(f"Disease Detected: {disease_label}")

if __name__ == "__main__":
    main()
