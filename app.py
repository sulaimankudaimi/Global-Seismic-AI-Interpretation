import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown # مكتبة لتحميل الملفات من درايف بسهولة

# إعدادات الواجهة
st.set_page_config(page_title="Seismic AI Interpreter", layout="wide")

st.markdown("<h1 style='text-align: center;'>🌊 Seismic Facies AI Interpreter</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>Developed by: Sulaiman Kudaimi</h3>", unsafe_allow_html=True)
st.divider()

# رابط الموديل الخاص بك على Google Drive (رابط التحميل المباشر)
MODEL_URL = 'https://drive.google.com/uc?id=1sbByP3UVgrm97hjziA2KyIxWQAlOL0eL'
MODEL_PATH = 'universal_seismic_model_v2.h5'

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('🚀 جاري تحميل الموديل الذكي لأول مرة (85MB)... يرجى الانتظار...'):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
    # تحميل الموديل مع تعريف دالة mse إذا لزم الأمر
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

try:
    model = load_model_from_drive()
    st.success("✅ تم تفعيل الذكاء الاصطناعي بنجاح!")
except Exception as e:
    st.error(f"❌ حدث خطأ أثناء جلب الموديل: {e}")

# منطقة رفع الملفات السيزمية
uploaded_file = st.file_uploader("ارفع ملف البيانات السيزمية (.dat)", type=["dat"])

if uploaded_file is not None:
    # قراءة البيانات وتحويلها لمصفوفة 128x128
    raw_bytes = uploaded_file.read()
    raw_data = np.frombuffer(raw_bytes, dtype=np.float32)
    
    if len(raw_data) >= 16384:
        img = raw_data[:16384].reshape((128, 128))
        
        # التوقع
        norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
        with st.spinner('🔍 جاري التحليل الجيولوجي...'):
            pred = model.predict(norm.reshape(1, 128, 128, 1), verbose=0)
        
        # عرض النتائج
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("البيانات الأصلية")
            st.image(norm, use_column_width=True, clamp=True)
        with col2:
            st.subheader("تفسير الذكاء الاصطناعي")
            st.image(pred[0,:,:,0], use_column_width=True, clamp=True)
    else:
        st.error("أبعاد الملف غير متوافقة. يرجى رفع ملف يحتوي على 16384 نقطة (128x128).")