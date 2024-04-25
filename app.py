import streamlit as st
import joblib
from PIL import Image
import numpy as np

# Load the pre-trained model
model = joblib.load("rj.joblib")

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size expected by the model
    resized_image = image.resize((600, 800))
    # Convert the image to a numpy array
    image_array = np.array(resized_image) / 255.0  # Normalize pixel values
    # Reshape the array to match the input shape expected by the model
    processed_image = image_array.reshape(1, 150, 200, 3)
    return processed_image

# Streamlit app
def main():
    # Title of the app
    st.title("Image Classification App")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Check if an image has been uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict button
        if st.button("Predict"):
            # Predict the class
            predicted_class = model.predict(processed_image)
            # Display the predicted class
            st.write(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()
