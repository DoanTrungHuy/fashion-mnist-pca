import glob
import pickle
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tensorflow.keras import datasets, layers, models
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define class labels for Fashion MNIST dataset
label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))  # Resize to the same dimensions as the training data
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img).astype('float32')
    img_array = img_array.reshape(1, -1)  # Reshape to 2D array
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        img_array = scaler.transform(img_array)
        
        with open('pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)
            img_array = pca.transform(img_array)
            return img_array
    except Exception as e:
        st.error("Failed to load PCA model: " + str(e))
        return None

working_dir = os.path.dirname(os.path.abspath(__file__))
data_train_path = f"{working_dir}/fashion_mnist_image_train"

images = []
labels = []

for file_name in os.listdir(data_train_path):
    # print("Readed")
    file_path = os.path.join(data_train_path, file_name)
    img = Image.open(file_path)
    img_array = np.array(img)
    images.append(img_array)
    label = int(file_name.split('.')[0][-1]) 
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

def min_distance(image_test, predicted_class):
    min_val = None
    min_dis = float('inf')
    n = len(images)
    for i in range(n):
        train_img_array = images[i]
        if predicted_class != labels[i]:
            continue
        train_img_array = train_img_array[:, :, 0]
        distance = euclidean_distances(train_img_array, image_test).sum()
        if min_dis > distance:
            min_dis = distance
            min_val = train_img_array
    return min_val

# Streamlit App
st.title('Fashion Item Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Input Image")

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image
            img_array = preprocess_image(uploaded_image)

            # Make a prediction using the pre-trained model
            result = tf.nn.softmax(model.predict(img_array))
            predicted_class = np.argmax(result)
            prediction = label_names[predicted_class]

            # st.write(str(result))
            img_array_dis = uploaded_image
            img = Image.open(img_array_dis)
            img = img.resize((28, 28))  # Resize to the same dimensions as the training data
            img = img.convert('L')  # Convert to grayscale
            img_array_dis = np.array(img).astype('float32')

            min_dis = min_distance(img_array_dis, predicted_class)
            st.image(min_dis, clamp=True, caption="Image from the training set during testing")
            
            st.success(f'Prediction: {prediction}')