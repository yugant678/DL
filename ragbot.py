import streamlit as st
import os, zipfile, shutil
from cnn_model import build_cnn
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import tempfile

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

st.set_page_config("📸 CNN Image Classifier", layout="wide")
st.title("🧠 CNN with Built-In Dataset Loader")

# Sidebar
st.sidebar.header("🔍 Choose Section")
section = st.sidebar.radio("Go to:", ["Train Model", "Predict"])

# Session state to hold model
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.img_size = 64
    st.session_state.class_names = []

if section == "Train Model":
    dataset_choice = st.sidebar.radio("Dataset:", ["CIFAR-10", "Cats vs Dogs (ZIP Upload)"])
    img_size = st.sidebar.slider("Image Size", 32, 256, 64)
    batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64])
    epochs = st.sidebar.slider("Epochs", 1, 20, 5)

    # CASE 1: CIFAR-10 Dataset
    if dataset_choice == "CIFAR-10":
        st.subheader("📦 CIFAR-10 Loaded")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train, x_test = x_train / 255.0, x_test / 255.0
        y_train_cat = to_categorical(y_train, 10)
        y_test_cat = to_categorical(y_test, 10)

        if st.button("🧠 Train CNN on CIFAR-10"):
            model = build_cnn((32, 32, 3), 10)
            history = model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat), epochs=epochs)

            st.session_state.model = model
            st.session_state.img_size = 32
            st.session_state.class_names = [str(i) for i in range(10)]

            st.success("✅ Training Complete!")
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(history.history["accuracy"], label="Train Acc")
            ax[0].plot(history.history["val_accuracy"], label="Val Acc")
            ax[0].set_title("Accuracy"); ax[0].legend()

            ax[1].plot(history.history["loss"], label="Train Loss")
            ax[1].plot(history.history["val_loss"], label="Val Loss")
            ax[1].set_title("Loss"); ax[1].legend()
            st.pyplot(fig)

    # CASE 2: Cats vs Dogs ZIP
    else:
        st.subheader("🐱🐶 Upload Cats vs Dogs ZIP")
        zip_file = st.file_uploader("Upload ZIP of dataset (foldered by class)", type=["zip"])
        if zip_file:
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            st.success("✅ Extracted!")

            datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
            train_gen = datagen.flow_from_directory(
                temp_dir, target_size=(img_size, img_size),
                batch_size=batch_size, class_mode='categorical', subset='training'
            )
            val_gen = datagen.flow_from_directory(
                temp_dir, target_size=(img_size, img_size),
                batch_size=batch_size, class_mode='categorical', subset='validation'
            )

            if st.button("🧠 Train CNN on Uploaded Dataset"):
                model = build_cnn((img_size, img_size, 3), train_gen.num_classes)
                history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

                st.session_state.model = model
                st.session_state.img_size = img_size
                st.session_state.class_names = list(train_gen.class_indices.keys())

                st.success("🎯 Training Finished!")
                fig, ax = plt.subplots(1, 2)
                ax[0].plot(history.history["accuracy"], label="Train Acc")
                ax[0].plot(history.history["val_accuracy"], label="Val Acc")
                ax[0].set_title("Accuracy"); ax[0].legend()

                ax[1].plot(history.history["loss"], label="Train Loss")
                ax[1].plot(history.history["val_loss"], label="Val Loss")
                ax[1].set_title("Loss"); ax[1].legend()
                st.pyplot(fig)

elif section == "Predict":
    st.subheader("🔎 Predict Image Class")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file and st.session_state.model is not None:
        img = load_img(uploaded_file, target_size=(st.session_state.img_size, st.session_state.img_size))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = st.session_state.model.predict(img_array)
        class_name = st.session_state.class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100

        st.success(f"✅ Predicted: {class_name} ({confidence:.2f}% confidence)")
    elif uploaded_file:
        st.warning("⚠️ Please train a model first in the 'Train Model' tab.")