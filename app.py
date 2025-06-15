import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# Load trained model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# Style
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#4CAF50;">🩺 AI Diabetes Predictor</h1>
        <p>Check your diabetes risk using AI — powered by a machine learning model trained on real health data.</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.success(result)
# 📊 Show bar chart of input values
features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
values = [glucose, blood_pressure, bmi, age]
normal_ranges = [120, 80, 25, 35]

fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(features))

ax.bar(index, values, bar_width, label='Your Values', color='skyblue')
ax.bar(index + bar_width, normal_ranges, bar_width, label='Normal Range', color='lightgreen')
ax.set_xlabel('Health Indicators')
ax.set_ylabel('Value')
ax.set_title('Health Indicator Comparison')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(features)
ax.legend()

st.pyplot(fig)

st.title("🩺 AI Diabetes Predictor")
st.write("Enter the values below to check the diabetes risk.")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
glucose = st.number_input("Glucose Level", 0, 200, step=1)
blood_pressure = st.number_input("Blood Pressure", 0, 150, step=1)
skin_thickness = st.number_input("Skin Thickness", 0, 100, step=1)
insulin = st.number_input("Insulin Level", 0, 900, step=1)
bmi = st.number_input("BMI", 0.0, 70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, step=0.01)
age = st.number_input("Age", 1, 120, step=1)

# Predict
if st.button("Predict Diabetes"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    result = "🔴 High Risk - Positive for Diabetes" if prediction[0] == 1 else "🟢 Low Risk - Negative for Diabetes"
    st.subheader("Prediction Result:")
    st.success(result)
