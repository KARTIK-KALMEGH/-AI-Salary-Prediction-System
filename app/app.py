import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("../model/salary_model.pkl")

st.title("💼 AI Salary Prediction System")
st.write("Predict Employee Expected Salary")

# User Inputs
total_exp = st.number_input("Total Experience (Years)", min_value=0)
current_ctc = st.number_input("Current CTC")
companies = st.number_input("Number of Companies Worked", min_value=0)
appraisal = st.number_input("Last Appraisal Rating", min_value=0.0)

if st.button("Predict Salary"):
    
    input_data = pd.DataFrame({
        "Total_Experience": [total_exp],
        "Current_CTC": [current_ctc],
        "No_Of_Companies_worked": [companies],
        "Last_Appraisal_Rating": [appraisal]
    })
    
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Expected Salary: ₹ {round(prediction[0],2)}")