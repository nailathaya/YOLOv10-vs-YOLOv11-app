import streamlit as st
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Judul aplikasi
st.title("Perbandingan Prediksi YOLOv10 vs YOLOv11")

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png'])

# Load model hanya sekali
@st.cache_resource
def load_models():
    yolov10 = YOLO("models/yolov10m/best.pt")
    yolov11 = YOLO("models/yolov11m/best.pt")
    return yolov10, yolov11

# Proses jika ada gambar yang diunggah
if uploaded_file is not None:
    # Tampilkan gambar asli
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang Diupload", use_container_width=True)

    # Load model
    yolov10, yolov11 = load_models()

    # Simpan gambar sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # Inference dengan kedua model
    result10 = yolov10(image_path)[0]
    result_img10 = result10.plot()

    result11 = yolov11(image_path)[0]
    result_img11 = result11.plot()

    # Tampilkan hasil berdampingan dengan caption
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("YOLOv10")
        st.image(result_img10, caption="Hasil Prediksi YOLOv10", use_container_width=True)
    with col2:
        st.subheader("YOLOv11")
        st.image(result_img11, caption="Hasil Prediksi YOLOv11", use_container_width=True)

    # Hapus file sementara
    os.remove(image_path)
