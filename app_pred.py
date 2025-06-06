import streamlit as st
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageOps
import tempfile
import os

# # Judul aplikasi
# st.title("Perbandingan Prediksi YOLOv10 vs YOLOv11")

# # Upload gambar
# uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png'])

# # Load model hanya sekali
# @st.cache_resource
# def load_models():
#     yolov10 = YOLO("models/yolov10m/best.pt")
#     yolov11 = YOLO("models/yolov11m/best.pt")
#     return yolov10, yolov11

# # Proses jika ada gambar yang diunggah
# if uploaded_file is not None:
#     # Tampilkan gambar asli
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption="Gambar yang Diupload", use_column_width=True)

#     # Load model
#     yolov10, yolov11 = load_models()

#     # Simpan gambar sementara
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#         image.save(tmp.name)
#         image_path = tmp.name

#     # Inference dengan kedua model
#     result10 = yolov10(image_path)[0]
#     result_img10 = result10.plot()

#     result11 = yolov11(image_path)[0]
#     result_img11 = result11.plot()

#     # Tampilkan hasil berdampingan dengan caption
#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("YOLOv10")
#         st.image(result_img10, caption="Hasil Prediksi YOLOv10", use_column_width=True)
#     with col2:
#         st.subheader("YOLOv11")
#         st.image(result_img11, caption="Hasil Prediksi YOLOv11", use_column_width=True)

#     # Hapus file sementara
#     os.remove(image_path)

# Judul aplikasi
st.title("Perbandingan Prediksi YOLOv10 vs YOLOv11")

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png'])

# Load model hanya sekali
@st.cache_resource
def load_models():
    """Loads the YOLO models."""
    yolov10 = YOLO("models/yolov10m/best.pt")
    yolov11 = YOLO("models/yolov11m/best.pt")
    return yolov10, yolov11

def preprocess_image(image):
    """
    Preprocesses the image by converting to grayscale and applying
    auto-adjust contrast using histogram equalization.
    """
    # 1. Konversi ke Grayscale
    gray_image = image.convert('L')

    # 2. Auto-Adjust Contrast menggunakan Histogram Equalization
    equalized_image = ImageOps.equalize(gray_image)
    
    # Konversi kembali ke RGB agar bisa di-render oleh model YOLO
    # Model YOLO mengharapkan input 3-channel
    preprocessed_image = equalized_image.convert('RGB')
    
    return preprocessed_image

# Proses jika ada gambar yang diunggah
if uploaded_file is not None:
    # Buka gambar
    original_image = Image.open(uploaded_file).convert('RGB')

    # Tampilkan gambar asli dan gambar yang sudah diproses
    st.subheader("Gambar Asli")
    st.image(original_image, caption="Gambar yang Diupload", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(original_image)
    
    st.subheader("Gambar Setelah Preprocessing")
    st.image(preprocessed_image, caption="Grayscale & Histogram Equalization", use_column_width=True)
    
    st.markdown("---")

    # Load model
    yolov10, yolov11 = load_models()

    # Simpan gambar yang sudah diproses sementara untuk inferensi
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        preprocessed_image.save(tmp.name)
        image_path = tmp.name

    # Inference dengan kedua model menggunakan gambar yang sudah diproses
    result10 = yolov10(image_path)[0]
    result_img10 = result10.plot() # plot() mengembalikan array numpy (BGR)

    result11 = yolov11(image_path)[0]
    result_img11 = result11.plot() # plot() mengembalikan array numpy (BGR)
    
    # Konversi hasil plot dari BGR (OpenCV default) ke RGB (Streamlit default)
    result_img10_rgb = cv2.cvtColor(result_img10, cv2.COLOR_BGR2RGB)
    result_img11_rgb = cv2.cvtColor(result_img11, cv2.COLOR_BGR2RGB)


    # Tampilkan hasil berdampingan dengan caption
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("YOLOv10")
        st.image(result_img10_rgb, caption="Hasil Prediksi YOLOv10", use_column_width=True)
    with col2:
        st.subheader("YOLOv11")
        st.image(result_img11_rgb, caption="Hasil Prediksi YOLOv11", use_column_width=True)

    # Hapus file sementara
    os.remove(image_path)