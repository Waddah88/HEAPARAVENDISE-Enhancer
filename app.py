import cv2
import numpy as np
import streamlit as st

# 1. إعداد الصفحة بالعرض الكامل واختيار الاسم الرسمي الفريد للمتصفح
st.set_page_config(
    page_title="Heaparavendise Enhancer", 
    page_icon="✨",
    layout="wide"
)

# 2. كود احترافي سري للغاية لحجب هويّة وإعلانات شركة Streamlit بالكامل (شريط علوي، أزرار، تذييل)
hide_branding_style = """
    <style>
    /* إخفاء القائمة العلوية وزر التطوير الافتراضي التابع للشركة */
    header {visibility: hidden !important;}
    .stDeployButton {display:none !important;}
    div[data-testid="stToolbar"] {visibility: hidden !important; height: 0px !important;}
    div[data-testid="stDecoration"] {visibility: hidden !important; height: 0px !important;}
    
    /* إخفاء جملة "Made with Streamlit" في أسفل الشاشة تماماً */
    footer {visibility: hidden !important;}
    .streamlit-footer {display: none !important;}
    
    /* إزالة الفراغات الميتة الناتجة عن حجب الشريط العلوي لملء الشاشة */
    .block-container {padding-top: 2rem !important; padding-bottom: 0rem !important;}
    </style>
"""
st.markdown(hide_branding_style, unsafe_allow_html=True)

# 3. واجهة النظام الرأسية النظيفة الحاملة لاسمك وشعارك التجاري الخاص فقط
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>✨ Heaparavendise Enhancer ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0A0A0; font-size: 1.2rem;'>النظام المطور لتصفية، معالجة، وتحسين جودة الصور بدقة فائقة</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)

# 4. محرك المعالجة المحسن (Image Processing Engine)
def enhance_image(input_img, contrast_level, noise_level, sharp_level):
    if input_img is None: 
        return None
    
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

# 5. توزيع الشاشة على عمودين متوازيين بالوضع العريض الاحترافي
col1, col2 = st.columns()

with col1:
    st.markdown("### 📥 مصدر المدخلات والتحكم")
    
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
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        result = enhance_image(opencv_image, contrast_slider, noise_slider, sharp_slider)
        st.image(result, caption="المعاينة الفورية للصورة المعدلة احترافياً", use_container_width=True)
        
        # محرك التنزيل الفوري لذاكرة الاستوديو
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
        st.info("💡 النظام في وضع الاستعداد. يرجى تزويد التطبيق بصورة لبدء المعالجة الذكية فوراً.")
