import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model
model = pickle.load(open(r'C:\Users\POOJA O\OneDrive\Desktop\HeartApp\rf_model.pkl','rb'))

st.title("Heart Attack Risk Classification App")

# Inputs
age = st.number_input("Age", 20, 100, 25)
restingbp = st.number_input("RestingBP", 0, 300, 100)
cholesterol = st.number_input("Cholesterol", 0, 700, 140)
fastingbs = st.selectbox("FastingBS", (0, 1))
maxhr = st.number_input("MaxHR", 60, 250, 140)
oldpeak = st.number_input("Oldpeak", -3.0, 6.6, 1.0)

gender = st.selectbox("Gender", ("M", "F"))
chestpaintype = st.selectbox("ChestPainType", ("TA", "ATA", "NAP", "ASY"))
restingecg = st.selectbox("RestingECG", ("Normal", "ST", "LVH"))
exerciseangina = st.selectbox("ExerciseAngina", ("Y", "N"))

# ✅ FIXED: correct variable name
st_slope = st.selectbox("ST_Slope", ("Up", "Flat", "Down"))

# Encoding
Exercise_Angina = 1 if exerciseangina == 'Y' else 0

Sex_F = 1 if gender == 'F' else 0
Sex_M = 1 if gender == 'M' else 0

Chest_PainType_dict = {'ATA':1, 'NAP':2, 'ASY':3, 'TA':4}
Chest_PainType = Chest_PainType_dict[chestpaintype]

Resting_ECG_dict = {'Normal':1, 'ST':2, 'LVH':3}
Resting_ECG = Resting_ECG_dict[restingecg]

St_Slope_dict = {'Up':1, 'Flat':2, 'Down':3}
ST_Slope = St_Slope_dict[st_slope]

# DataFrame
input_features = pd.DataFrame({
    'Age':[age],
    'RestingBP':[restingbp],
    'Cholesterol':[cholesterol],
    'FastingBS':[fastingbs],
    'MaxHR':[maxhr],
    'Oldpeak':[oldpeak],
    'Exercise_Angina':[Exercise_Angina],
    'Sex_F':[Sex_F],
    'Sex_M':[Sex_M],
    'Chest_PainType':[Chest_PainType],
    'Resting_ECG':[Resting_ECG],
    'st_Slope':[ST_Slope]
})

# ⚠️ TEMP scaler (better to use saved scaler)
scaler = StandardScaler()
input_features[['Age','RestingBP','Cholesterol','MaxHR']] = scaler.fit_transform(
    input_features[['Age','RestingBP','Cholesterol','MaxHR']]
)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_features)[0]

    if prediction == 1:
        st.error("High Heart Attack Risk")
    else:
        st.success("Low Heart Attack Risk")