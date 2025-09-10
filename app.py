import streamlit as st
import pandas as pd
import joblib
import io

# Load your trained model, scaler, and expected columns
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.set_page_config(page_title="❤️ Heart Stroke Prediction", page_icon="❤️", layout="centered")

st.title("❤️ Heart Stroke Prediction by Sarthak")
st.markdown("Fill in your details to check your risk of heart disease:")

# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 250, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prediction button
if st.button("🔍 Predict"):

    # Build input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
        result_text = "High Risk of Heart Disease"
    else:
        st.success("✅ Low Risk of Heart Disease")
        result_text = "Low Risk of Heart Disease"

    # -------------------------
    # ✅ Download button appears here
    # -------------------------
    report = f"""
    🧑 Age: {age}
    ⚧ Sex: {sex}
    ❤️ Chest Pain: {chest_pain}
    💉 Cholesterol: {cholesterol}
    🫀 MaxHR: {max_hr}
    📉 Oldpeak: {oldpeak}

    ✅ Prediction: {result_text}
    """

    buffer = io.BytesIO(report.encode("utf-8"))

    st.download_button(
        label="📥 Download Report",
        data=buffer,
        file_name="heart_stroke_report.txt",
        mime="text/plain"
    )
