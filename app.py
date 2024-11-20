import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('mriclassifier_model.h5')

# Define the class labels
CLASS_LABELS = ['Healthy', 'Glioma', 'Meningioma', 'Pituitary']

# Streamlit App Title
st.title("MRI Brain Tumor Classifier")

# File Uploader for MRI Image
uploaded_file = st.file_uploader("Upload MRI Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Resize the image to match the model's input size
    image_resized = image.resize((128, 128))

    # Convert grayscale images to RGB
    if image_resized.mode != "RGB":
        image_resized = image_resized.convert("RGB")

    # Convert the image to a NumPy array and normalize pixel values
    image_array = np.array(image_resized) / 255.0  # Normalize to [0, 1]

    # Add a batch dimension (1, 128, 128, 3)
    image_array = np.expand_dims(image_array, axis=0)

    # Display the image
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Make a prediction
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = CLASS_LABELS[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100  # Confidence as a percentage

    # Display Prediction and Confidence
    st.write(f"### Prediction: {predicted_class_label}")
    st.write(f"### Confidence: {confidence:.2f}%")

    # Debugging output
    st.write("Prediction probabilities for each class:")
    for label, prob in zip(CLASS_LABELS, prediction[0]):
        st.write(f"{label}: {prob * 100:.2f}%")
