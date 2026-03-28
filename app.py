# Gender -> 1 Female 0 Male
# Churn -> 1  Yes 0 No 
#Scaler is exported as scaler.pkl
#Model is exported as best_model.pkl
#Oredr of the x -> 'Age','Gender','Tenure','MonthlyCharges','TotalCharges'

import streamlit as st
import joblib
import numpy as np
scaler = joblib.load("scaler.pkl")
model = joblib.load("best_model.pkl")

st.title("Customer Churn Prediction App")
st.divider()
st.write("Please enter the values and hit the predict button for getting a prediction.")
st.divider()
age=st.number_input("enter age",min_value=0,max_value=100,value=10)

gender=st.selectbox("select gender",["Male","Female"])

tenure = st.number_input("enter tenure",min_value=0,max_value=130,value=10)
monthly_charges = st.number_input("enter monthly charges",min_value=30,max_value=150)
gender=st.selectbox("Enter the Gender",["Male","Female"])
st.divider()
predictbutton=st.button("Predict!")
st.divider()
if predictbutton:
    gender_selected = 1 if gender == "Female" else 0
    x = [age,gender_selected,tenure,monthly_charges]
    x1 = np.array(x)
    x_array=scaler.transform([x1])
    prediction = model.predict(x_array)[0]
    predicted= "Yes" if prediction == 1 else "No"
    st.balloons()
    st.write(f"The predicted class is : {predicted}")


else:
    st.write("Please enter the values and use predict button .")
