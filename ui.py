import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. إعداد شكل الصفحة (خليناها Wide عشان تاخد الشاشة كلها)
st.set_page_config(page_title="Car Price Predictor", page_icon="🏎️", layout="wide")

# ==========================================
# 2. كود CSS لتجميل الواجهة (عشان الشكل يبقى احترافي)
# ==========================================
st.markdown("""
    <style>
    .main-title {
        font-size: 45px;
        color: #ff4b4b;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 20px;
        color: #808495;
        text-align: center;
        margin-bottom: 40px;
    }
    .result-card {
        background-color: #2ecb71;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 35px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# 3. تحميل الموديل والملفات
@st.cache_resource
def load_models():
    model = joblib.load('xgb_final_model.pkl')
    brand_encoder = joblib.load('brand_encoder.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, brand_encoder, model_columns

model, brand_encoder, model_columns = load_models()

# ==========================================
# 4. القائمة الجانبية (Sidebar) - بتدي انطباع احترافي
# ==========================================
with st.sidebar:
    # حطيت رابط لصورة عربية أيكون شكلها حلو
    st.image("https://cdn-icons-png.flaticon.com/512/3202/3202926.png", width=150)
    st.header("⚙️ عن المشروع")
    st.info("هذا النظام يعتمد على خوارزمية **XGBoost** لتوقع أسعار السيارات المستعملة بدقة عالية بناءً على مواصفاتها.")
    st.markdown("---")
    st.markdown("👨‍💻 **فريق العمل:**")
    st.markdown("- Youssef Mohamed\n-Mahmoud Shawky\n-Anne-Marei Josaph\n-Malak Mahmoud\n-Yara Mustafa")

# ==========================================
# 5. واجهة المستخدم الرئيسية (UI)
# ==========================================
st.markdown('<div class="main-title">🏎️ نظام التوقع الذكي لأسعار السيارات</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">أدخل مواصفات سيارتك بدقة للحصول على تقييم فوري لسعرها في السوق</div>', unsafe_allow_html=True)
st.markdown("---")

# تقسيم الشاشة لـ 3 عواميد
col1, col2, col3 = st.columns(3)

available_brands = sorted(list(brand_encoder.keys()))

with col1:
    st.subheader("📋 المواصفات الأساسية")
    brand = st.selectbox("ماركة العربية", available_brands)
    year = st.number_input("سنة الصنع", min_value=1980, max_value=2026, value=2018)
    km_driven = st.number_input("المسافة المقطوعة (كم)", min_value=0, value=45000, step=1000)

with col2:
    st.subheader("⚙️ مواصفات المحرك")
    engine = st.number_input("سعة المحرك (CC)", min_value=0.0, value=1248.0, step=100.0)
    max_power = st.number_input("قوة المحرك (bhp)", min_value=0.0, value=74.0, step=5.0)
    mileage = st.number_input("استهلاك الوقود (kmpl)", min_value=0.0, value=23.4, step=1.0)

with col3:
    st.subheader("🚘 تفاصيل أخرى")
    fuel = st.selectbox("نوع الوقود", ["Diesel", "Petrol", "CNG", "LPG"])
    transmission = st.selectbox("ناقل الحركة", ["Manual", "Automatic"])
    seller_type = st.selectbox("نوع البائع", ["Individual", "Dealer", "Trustmark Dealer"])
    owner = st.selectbox("عدد الملاك السابقين", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

st.markdown("---")

# توسيط زرار التوقع
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🚀 اضغط هنا لتوقع السعر 🚀", use_container_width=True)

# ==========================================
# 6. معالجة البيانات وعرض النتيجة
# ==========================================
if predict_btn:
    try:
        car_age = 2024 - year
        brand_encoded = brand_encoder.get(brand, np.mean(list(brand_encoder.values())))
        
        input_dict = {
            'km_driven': km_driven,
            'car_age': car_age,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'brand_encoded': brand_encoded,
            f'fuel_{fuel}': 1,
            f'seller_type_{seller_type}': 1,
            f'transmission_{transmission}': 1,
            f'owner_{owner}': 1
        }
        
        input_data = pd.DataFrame([input_dict])
        
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0
                
        input_data = input_data[model_columns]
        
        prediction = model.predict(input_data)[0]
        
        # عرض النتيجة جوه الكارت الأخضر اللي صممناه بالـ CSS
        st.markdown(f'<div class="result-card">💰 السعر المتوقع: {prediction:,.0f} روبية (INR)</div>', unsafe_allow_html=True)
        st.balloons()
        
    except Exception as e:
        st.error(f"حصلت مشكلة في التوقع: {e}")