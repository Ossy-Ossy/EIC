import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

st.write("# Welcome to the **UNN startup business** sucess rate prediction app")
st.write("***")
x = pd.read_csv("x_startup.csv")
age = x.columns[0:5]
gender = x.columns[5:7]
department = x.columns[7:19]
level = x.columns[19:26]
business = x.columns[26:55]
longetivity = x.columns[55:59]
entrepreneur_programs = x.columns[59:61]
mentorship = x.columns[61:63]
funding = x.columns[63:72]
balance = x.columns[72:]

model = load('model_EIC')

def prediction(x ,model ,*columns):
    fv = np.zeros(len(x.columns))
    for cols in columns:
        index = x.columns.get_loc(cols)
        fv[index] = 1
        prediction_result = model.predict(fv.reshape(1 ,-1))
        probability =  model.predict_proba(fv.reshape(1 ,-1)) * 100
    if prediction_result == 0:
        return(f"The business is guaranteed not to be successful and earn you below N50,000 monthly with a probability of {probability[0][0]} %")
    else:
        return(f"The business is guaranteed to be successful and earn you above N50,000 monthly with a probability of {probability[0][1] }%")
    return prediction_result
st.subheader("Select your plans for the business")
st.write('***')
age_ch= st.selectbox("What is your age" ,age) 
gender_ch = st.selectbox("What is your gender" ,gender)
department_ch = st.selectbox("What department are you from?" ,department)
level_ch = st.selectbox("What level are you in currently?" ,level)
st.write('***')
business_ch = st.selectbox("What type of business are you into?" ,business)
longetivity_ch = st.selectbox("How long have your business been running?", longetivity)
entrepreneur_programs_ch = st.selectbox("Do you attend entrepreneurship oriented programs?" ,entrepreneur_programs)
mentorship_ch = st.selectbox("Do you have any mentor to your business?" ,mentorship)
st.write('***')
funding_ch = st.selectbox("What is your means of funding your business?" ,funding)
balance_ch = st.selectbox("What is your challenge in your business" ,balance)

if st.button("predict"):
    sucess_rate = prediction(x ,model ,age_ch,gender_ch,department_ch,level_ch,business_ch,longetivity_ch,entrepreneur_programs_ch,mentorship_ch,funding_ch,balance_ch)
    st.success(f"prediction : {sucess_rate}")