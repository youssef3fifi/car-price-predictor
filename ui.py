import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from fpdf import FPDF

# منع التحذيرات بتاعة مكتبات الرسم
st.set_option('deprecation.showPyplotGlobalUse', False)

# 1. إعداد شكل الصفحة
st.set_page_config(page_title="Car Price Predictor", page_icon="🏎️", layout="wide")

# 2. كود CSS لتجميل الواجهة
st.markdown("""
    <style>
    .main-title {font-size: 45px; color: #ff4b4b; text-align: center; font-weight: bold; margin-bottom: 10px;}
    .sub-title {font-size: 20px; color: #808495; text-align: center; margin-bottom: 40px;}
    .result-card {background-color: #2ecb71; padding: 20px; border-radius: 10px; text-align: center; color: white; font-size: 35px; font-weight: bold; margin-top: 20px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);}
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

# 4. القائمة الجانبية (Sidebar)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3202/3202926.png", width=150)
    st.header("⚙️ عن المشروع")
    st.info("نظام ذكي متكامل يعتمد على XGBoost لتوقع أسعار السيارات، مزود بشهادات تسعير وتحليل بيانات.")
    st.markdown("---")
    st.markdown("👨‍💻 **فريق العمل:**")
    st.markdown("- Youssef Mohamed\n- Mahmoud Shawky\n- Anne-Marei Josaph\n- Malak Mahmoud\n- Yara Mustafa")

st.markdown('<div class="main-title">🏎️ نظام التوقع الذكي لأسعار السيارات</div>', unsafe_allow_html=True)

# ==========================================
# إضافة ميزة التبويبات (Tabs) لفصل التوقع عن لوحة البيانات
# ==========================================
tab1, tab2 = st.tabs(["🚀 توقع السعر (Predictor)", "📊 لوحة بيانات السوق (Market Dashboard)"])

with tab1:
    st.markdown('<div class="sub-title">أدخل مواصفات سيارتك بدقة للحصول على تقييم فوري لسعرها في السوق</div>', unsafe_allow_html=True)
    st.markdown("---")

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

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("🚀 اضغط هنا لتوقع السعر 🚀", use_container_width=True)

    if predict_btn:
        try:
            car_age = 2024 - year
            brand_encoded = brand_encoder.get(brand, np.mean(list(brand_encoder.values())))
            
            input_dict = {
                'km_driven': km_driven, 'car_age': car_age, 'mileage': mileage,
                'engine': engine, 'max_power': max_power, 'brand_encoded': brand_encoded,
                f'fuel_{fuel}': 1, f'seller_type_{seller_type}': 1,
                f'transmission_{transmission}': 1, f'owner_{owner}': 1
            }
            
            input_data = pd.DataFrame([input_dict])
            for col in model_columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[model_columns]
            
            prediction_inr = model.predict(input_data)[0]
            inr_to_usd = 0.012
            inr_to_egp = 0.57
            prediction_usd = prediction_inr * inr_to_usd
            prediction_egp = prediction_inr * inr_to_egp
            
            st.markdown(f"""
                <div class="result-card">
                    💰 السعر المتوقع:
                    <hr style="margin: 10px 0; border-color: rgba(255,255,255,0.2);">
                    <div style="font-size: 22px; margin-bottom: 5px;">🇮🇳 {prediction_inr:,.0f} روبية هندية</div>
                    <div style="font-size: 26px; margin-bottom: 5px; color: #f1c40f;">🇺🇸 {prediction_usd:,.0f} دولار أمريكي</div>
                    <div style="font-size: 32px; font-weight: 900; color: #fff;">🇪🇬 {prediction_egp:,.0f} جنيه مصري</div>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()

            # --- البونص 1: الذكاء الاصطناعي المفسر (SHAP) ---
            st.markdown("### 🧠 كيف حسب الذكاء الاصطناعي هذا السعر؟")
            st.info("الرسم التالي يوضح العوامل التي أدت لزيادة أو نقصان سعر هذه السيارة تحديداً.")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.summary_plot(shap_values, input_data, plot_type="bar", show=False, color="#ff4b4b")
            st.pyplot(fig)

            # --- البونص 2: استخراج فاتورة PDF ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 20)
            pdf.cell(200, 15, txt="Official Car Valuation Certificate", ln=True, align='C')
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Generated by XGBoost AI Engine", ln=True, align='C')
            pdf.line(10, 35, 200, 35)
            pdf.ln(20)
            
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(100, 10, txt=f"Brand: {brand}", ln=True)
            pdf.cell(100, 10, txt=f"Manufacturing Year: {year}", ln=True)
            pdf.cell(100, 10, txt=f"Kilometers Driven: {km_driven:,.0f} km", ln=True)
            pdf.cell(100, 10, txt=f"Engine Specs: {engine} CC / {max_power} bhp", ln=True)
            pdf.cell(100, 10, txt=f"Transmission: {transmission} | Fuel: {fuel}", ln=True)
            
            pdf.ln(15)
            pdf.set_fill_color(46, 203, 113) # لون أخضر
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 15, txt=f" Estimated Value: {prediction_egp:,.0f} EGP", ln=True, align='C', fill=True)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            
            st.markdown("### 📄 طباعة التقرير الرسمي")
            st.download_button(label="📥 تحميل شهادة التسعير (PDF)", data=pdf_bytes, file_name="Car_Valuation_Certificate.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"حصلت مشكلة في التوقع: {e}")

# ==========================================
# البونص 3: لوحة بيانات تفاعلية (Dashboard) في التبويب الثاني
# ==========================================
with tab2:
    st.subheader("📊 إحصائيات الذكاء الاصطناعي (أهم العوامل المؤثرة عامةً)")
    st.write("الرسم البياني التالي يوضح أكثر المواصفات التي يعتمد عليها الموديل لتسعير أي سيارة في السوق.")
    
    # استخراج أهمية الميزات من الموديل
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [model_columns[i] for i in indices[:10]]
    top_importances = importances[indices][:10]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette="mako", ax=ax2)
    ax2.set_title("Top 10 Most Important Features in Market Pricing")
    ax2.set_xlabel("Importance Score")
    ax2.set_ylabel("Feature Name")
    st.pyplot(fig2)
    
    st.success("💡 ملاحظة: قوة المحرك (max_power) وعمر السيارة (car_age) هم المتحكمين الأساسيين في رفع أو خفض سعر السيارة في السوق.")
