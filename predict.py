"""import joblib
import mysql
import pandas as pd
import streamlit as st
import mysql.connector

st.text("welcome")
st.write("hello")
knn=joblib.load('knn_model.pkl')
svr=joblib.load('svr_model.pkl')
nnm=joblib.load('neural_net_model.pkl')

temp=st.number_input("input the temperature",format="%.2f")
cat=st.number_input("input the Catalyst Concentration",format="%.2f")
pres=st.number_input("input the pressure",format="%.2f")

def run_model(temp,cat,pres,a):
    data=[[temp,pres,cat]]
    if a=='svr':
        return svr.predict(data)



a=st.selectbox("which ML you want to use",['select a option','svr',"nnm",'knn'])
st.write(f"you selected {a}")
if (a!='select a option'):
    if st.button('Run Prediction'):
        result=run_model(temp,cat,pres,a)
        st.write(f'Result :{result}')
    

"""

#-------------------------------------------------------------------------------------------------------------

import streamlit as st
import joblib
import numpy as np

# -------------------- Page Config --------------------
st.set_page_config(page_title="Chemical Yield Predictor", layout="centered")

# -------------------- Load Models and Scaler --------------------
knn = joblib.load('knn_model.pkl')
svr = joblib.load('svr_model.pkl')
nn = joblib.load('neural_net_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the correct scaler used during training

# -------------------- UI --------------------
st.markdown("# 🧪 Chemical Yield Predictor")
st.markdown("Predict chemical reaction yield (%) using trained ML models.")

with st.sidebar:
    st.title("📌 Instructions")
    st.markdown("""
    1. Enter experimental values below.
    2. Choose a machine learning model.
    3. Click 'Run Prediction' to get the estimated yield.
    """)

# -------------------- Inputs --------------------
st.subheader("🔢 Experimental Conditions")
temp = st.number_input("🌡️ Temperature (°C)", min_value=20.0, max_value=120.0, value=60.0, format="%.2f")
pres = st.number_input("⛽ Pressure (atm)", min_value=1.0, max_value=9.0, value=5.0, format="%.2f")
cat = st.number_input("🧪 Catalyst Concentration (mol/L)", min_value=0.05, max_value=1.0, value=0.5, format="%.2f")


st.subheader("🤖 Select Machine Learning Model")
model_name = st.selectbox("Choose model", ['Select a model', 'KNN', 'SVR', 'Neural Net'])

# -------------------- Prediction --------------------
def predict(temp, pres, cat, model_name):
    input_scaled = scaler.transform([[temp, pres, cat]])
    if model_name == 'KNN':
        return knn.predict(input_scaled)[0]
    elif model_name == 'SVR':
        return svr.predict(input_scaled)[0]
    elif model_name == 'Neural Net':
        return nn.predict(input_scaled)[0]

if model_name != 'Select a model':
    if st.button("🚀 Run Prediction"):
        result = predict(temp, pres, cat, model_name)
        st.success(f"🎯 Predicted Yield: {result:.2f}%")
