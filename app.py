import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="أداة تحسين وتصفية الصور المتقدمة", layout="centered")
st.markdown("<h2 style='text-align: center;'>أداة تحسين وتصفية الصور الاحترافية</h2>", unsafe_allow_html=True)

def enhance_image(input_img, contrast_level, noise_level, sharp_level):
    if input_img is None: return None
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLEHE(clipLimit=float(contrast_level), tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_contrast = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    if noise_level > 0:
        img_denoised = cv2.fastNlMeansDenoisingColored(img_contrast, None, noise_level, noise_level, 7, 21)
    else:
        img_denoised = img_contrast
    if sharp_level > 0:
        blurred = cv2.GaussianBlur(img_denoised, (5, 5), 1.0)
        img_sharp = cv2.addWeighted(img_denoised, 1.0 + float(sharp_level), blurred, -float(sharp_level), 0)
    else:
        img_sharp = img_denoised
    return cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB)

uploaded_file = st.camera_input("التقط صورة بكاميرا جوالك")

contrast_slider = st.slider("1) مستوى التباين والإضاءة (Contrast)", 0.0, 10.0, 2.0, 0.1)
noise_slider = st.slider("2) مستوى إزالة التشويش (Noise Reduction)", 0.0, 30.0, 0.0, 1.0)
sharp_slider = st.slider("3) مستوى حدة الصورة (Sharpness)", 0.0, 5.0, 0.0, 0.1)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    st.subheader("الصورة بعد التصفية والتحسين:")
    result = enhance_image(opencv_image, contrast_slider, noise_slider, sharp_slider)
    st.image(result, use_column_width=True)
