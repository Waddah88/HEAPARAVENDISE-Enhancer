import cv2
import numpy as np
import streamlit as st

# 1. إعداد الصفحة ليتم تشغيلها بالعرض الكامل (Wide Layout) بالاسم الجديد
st.set_page_config(
    page_title="Heaparavendise Enhancer", 
    page_icon="✨",
    layout="wide"
)

# عنوان التطبيق الرئيسي الاحترافي متموضع في المنتصف
st.markdown("<h1 style='text-align: center;'>✨ Heaparavendise Enhancer ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>أداة احترافية متقدمة لتصفية وتحسين جودة الصور بدقة عالية</p>", unsafe_allow_html=True)
st.write("---")

def enhance_image(input_img, contrast_level, noise_level, sharp_level):
    if input_img is None: return None
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # تحسين التباين
    clahe = cv2.createCLEHE(clipLimit=float(contrast_level), tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_contrast = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    # تقليل التشويش
    if noise_level > 0:
        img_denoised = cv2.fastNlMeansDenoisingColored(img_contrast, None, noise_level, noise_level, 7, 21)
    else:
        img_denoised = img_contrast
        
    # زيادة حدة الصورة وتوضيح الملامح
    if sharp_level > 0:
        blurred = cv2.GaussianBlur(img_denoised, (5, 5), 1.0)
        img_sharp = cv2.addWeighted(img_denoised, 1.0 + float(sharp_level), blurred, -float(sharp_level), 0)
    else:
        img_sharp = img_denoised
        
    return cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB)

# تقسيم واجهة المستخدم الرسومية إلى عمودين للاستفادة الكاملة من الشاشة الكبيرة
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📥 مدخلات الصورة")
    
    # إضافة خيار التبديل بين الكاميرا والاستوديو
    source_option = st.radio("اختر مصدر الصورة المناسب لك:", ("رفع صورة من الاستوديو (ملف)", "التقاط صورة حية بكاميرا الجوال"))
    
    uploaded_file = None
    if source_option == "رفع صورة من الاستوديو (ملف)":
        uploaded_file = st.file_uploader("قم باختيار صورة من جهازك..", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("التقط صورة مباشرة من هاتفك")
        
    st.write("---")
    st.subheader("🎛 لوحة تحكم الفلاتر")
    contrast_slider = st.slider("1) مستوى التباين والإضاءة (Contrast)", 0.0, 10.0, 2.0, 0.1)
    noise_slider = st.slider("2) مستوى إزالة التشويش (Noise Reduction)", 0.0, 30.0, 0.0, 1.0)
    sharp_slider = st.slider("3) مستوى حدة الصورة (Sharpness)", 0.0, 5.0, 0.0, 0.1)

with col2:
    st.subheader("🖥 معاينة النتيجة الاحترافية")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        # معالجة وعرض مخرجات الفلتر المطور
        result = enhance_image(opencv_image, contrast_slider, noise_slider, sharp_slider)
        st.image(result, caption="الصورة المعدلة بعد التصفية والتحسين المتقدم", use_container_width=True)
    else:
        st.info("💡 في انتظار تزويد التطبيق بصورة لبدء تشغيل محرك المعالجة المتقدم تلقائياً.")
