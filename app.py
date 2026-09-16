import cv2
import numpy as np
import streamlit as st

# 1. إعداد الصفحة بالعرض الكامل والاسم الجديد والأيقونة الاحترافية
st.set_page_config(
    page_title="Heaparavendise Enhancer", 
    page_icon="✨",
    layout="wide"
)

# 2. تصميم الواجهة الرأسية الاحترافية باستخدام HTML/CSS مدمج
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>✨ Heaparavendise Enhancer ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0A0A0; font-size: 1.2rem;'>النظام المطور لتصفية، معالجة، وتحسين جودة الصور بدقة فائقة</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)

# 3. محرك المعالجة الأساسي المحسن (Image Processing Engine)
def enhance_image(input_img, contrast_level, noise_level, sharp_level):
    if input_img is None: 
        return None
    
    # تحويل الصورة إلى بيئة BGR لـ OpenCV
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    
    # معالجة التباين الذكي (Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLEHE(clipLimit=float(contrast_level), tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_contrast = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    # محرك تصفية وإزالة التشويش الحركي الرقمي
    if noise_level > 0:
        img_denoised = cv2.fastNlMeansDenoisingColored(img_contrast, None, noise_level, noise_level, 7, 21)
    else:
        img_denoised = img_contrast
        
    # زيادة حدة الملامح (High-Pass Sharpness Filter)
    if sharp_level > 0:
        blurred = cv2.GaussianBlur(img_denoised, (5, 5), 1.0)
        img_sharp = cv2.addWeighted(img_denoised, 1.0 + float(sharp_level), blurred, -float(sharp_level), 0)
    else:
        img_sharp = img_denoised
        
    # إعادة الصورة إلى صيغة RGB الافتراضية للعرض
    return cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB)

# 4. تقسيم لوحة التحكم والمعاينة إلى عمودين متوازيين (Wide Layout Mode)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📥 مصدر المدخلات والتحكم")
    
    # خيار التبديل الذكي بين الاستوديو والكاميرا الحية
    source_option = st.radio(
        "اختر طريقة تزويد الصورة:", 
        ("رفع ملف من الاستوديو (الالبوم)", "التقاط صورة حية بواسطة الكاميرا")
    )
    
    uploaded_file = None
    if source_option == "رفع ملف من الاستوديو (الالبوم)":
        uploaded_file = st.file_uploader("اختر صورة مدعومة (PNG, JPG, JPEG):", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("وجه الكاميرا والتقط لقطة:")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎛 فلاتر التعديل والمعايرة")
    contrast_slider = st.slider("1) مستوى التباين والإضاءة (Contrast)", 0.0, 10.0, 2.0, 0.1)
    noise_slider = st.slider("2) مستوى تصفية التشويش (Noise Reduction)", 0.0, 30.0, 0.0, 1.0)
    sharp_slider = st.slider("3) مستوى حدة الملامح (Sharpness)", 0.0, 5.0, 0.0, 0.1)

with col2:
    st.markdown("### 🖥 شاشة المعاينة وحفظ النتائج")
    
    if uploaded_file is not None:
        # قراءة وتفكيك مصفوفة البايتات للصورة المرفوعة
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        # تشغيل الفلتر المطور واستخراج النتيجة
        result = enhance_image(opencv_image, contrast_slider, noise_slider, sharp_slider)
        
        # عرض النتيجة بالتحجيم المناسب المعتمد برمجياً
        st.image(result, caption="المعاينة الفورية للصورة المعدلة احترافياً", use_container_width=True)
        
        # 5. محرك الحفظ والتنزيل المباشر إلى ذاكرة الجهاز بجودة كاملة
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        img_bytes = img_encoded.tobytes()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="💾 تحميل وحفظ الصورة في الاستوديو",
            data=img_bytes,
            file_name="Heaparavendise_Enhanced.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    else:
        st.info("💡 النظام في وضع الاستعداد. يرجى تزويد التطبيق بصورة من القائمة الجانبية لبدء المعالجة الذكية فوراً.")
