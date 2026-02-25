import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

import streamlit as st
import requests
from utils.prediction import predict_diabetes
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Prevent.AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ---------------- SIDEBAR ----------------
# ---------------- SIDEBAR MODEL INFO ----------------
with st.sidebar:

    st.markdown("""
    <h2 style='margin-bottom: 0;'>🩺 PreventAI</h2>
    <p style='margin-top: 0; font-size: 13px; opacity: 0.8;'>
    AI-Based Diabetes Risk Prediction & Prevention System
    </p>
    <hr>
    """, unsafe_allow_html=True)
st.sidebar.markdown("## 📊 Model Overview")

st.sidebar.markdown("**Algorithm:** Logistic Regression")
st.sidebar.markdown("**Accuracy:** 87%")
st.sidebar.markdown("**Dataset:** PIMA Indians Diabetes Dataset")
st.sidebar.markdown("**Training Samples:** 768 records")
st.sidebar.markdown("**Features Used:** 8 Clinical Parameters")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔬 Features")
st.sidebar.markdown("""
- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🚀 Project Info")
st.sidebar.markdown("Team: **NextCore**")
st.sidebar.markdown("Theme: **HealthTech**")
st.sidebar.markdown("Version: 1.0 (Prototype)")

# ---------------- GLOBAL STYLES ----------------
st.markdown("""
<style>

/* Hide default top padding */
.block-container {
    padding-top: 3rem;
}

/* Place PreventAI in the main header zone */
header[data-testid="stHeader"]::after {
    content: "PreventAI";
    position: absolute;
    left: 70px;
    top: 12px;
    font-size: 20px;
    font-weight: 600;
    color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)
# ---------------- LOAD MODEL ----------------
model = joblib.load("model/diabetes_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.hero {
    background: linear-gradient(90deg, #1B4F72, #2E86C1);
    padding: 15px 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}

.hero h1 {
    font-size: 32px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 16px;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)



# ---------------- MAIN TWO-COLUMN LAYOUT ----------------
left_col, right_col = st.columns([1.2, 1])

# ================= LEFT SIDE =================
with left_col:

    st.subheader("🔍 Diabetes Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies", min_value=0, value=1)
        glucose = st.number_input("Glucose", value=120.0)
        bp = st.number_input("Blood Pressure", value=70.0)
        skin = st.number_input("Skin Thickness", value=20.0)

    with col2:
        insulin = st.number_input("Insulin", value=79.0)
        bmi = st.number_input("BMI", value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", value=0.5)
        age = st.number_input("Age", value=30)

    predict = st.button("Predict Risk")

    if predict:
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if probability < 0.3:
            level = "Low Risk"
            color = "green"
        elif probability < 0.7:
            level = "Moderate Risk"
            color = "orange"
        else:
            level = "High Risk"
            color = "red"

        st.session_state["user_data"] = {
            "Age": age,
            "BMI": bmi,
            "Glucose": glucose,
            "BloodPressure": bp,
            "Insulin": insulin,
            "DiabetesPedigreeFunction": dpf,
            "RiskProbability": round(probability * 100, 2),
            "RiskLevel": level
        }

        st.markdown(f"### {level} ({round(probability*100,2)}%)")

    # ---------- BULK CSV ----------
    st.markdown("---")
    st.subheader("📂 Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Outcome" in df.columns:
            df = df.drop("Outcome", axis=1)

        expected_columns = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]

        df = df[expected_columns]

        df_scaled = scaler.transform(df)
        df["Prediction"] = model.predict(df_scaled)
        df["RiskProbability (%)"] = model.predict_proba(df_scaled)[:,1] * 100

        st.success("Bulk Prediction Complete")
        st.dataframe(df)

# ================= RIGHT SIDE =================
with right_col:

    st.subheader("💬 AI Health Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello 👋 I'm your PreventAI assistant. How can I help you today?"}
        ]
 

    # Scrollable container (native Streamlit)
    chat_container = st.container(height=350)

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input OUTSIDE container
    prompt = st.chat_input("Ask about diabetes or your health risk...")

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        # Prepare system prompt
        if "user_data" in st.session_state:
            user_info = st.session_state["user_data"]
            system_prompt = f"""
You are a healthcare AI assistant.

Age: {user_info['Age']}
BMI: {user_info['BMI']}
Glucose: {user_info['Glucose']}
Risk Level: {user_info['RiskLevel']}
Risk Probability: {user_info['RiskProbability']}%

Provide personalized preventive advice.
Do not diagnose.
Be supportive and clear.
"""
        else:
            system_prompt = """
You are a healthcare AI assistant.
Provide general diabetes prevention advice.
Do not diagnose.
Be supportive and clear.
"""

        try:
            api_key = st.secrets["OPENROUTER_API_KEY"]

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                }
            )

            result = response.json()

            if "choices" in result:
                reply = result["choices"][0]["message"]["content"]
            else:
                reply = "Sorry, I couldn't process that."

        except:
            reply = "AI service temporarily unavailable."

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

        st.rerun()


    st.caption("⚠ PreventAI provides preventive health insights and does not replace medical diagnosis.")
    
