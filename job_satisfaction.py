import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load models and feature columns using pickle
with open('.venv\Include\job_satisfaction_dt.pkl', 'rb') as j:
    dt_model = pickle.load(j)

with open('.venv\Include\job_satisfaction_ln.pkl', 'rb') as m:
    lr_model = pickle.load(m)

with open('.venv\Include\job_satisfying.pkl', 'rb') as u:
    feature_columns = pickle.load(u)

# Streamlit app
st.title('Employee Job Satisfaction Predictor')

# Model selection
model_type = st.sidebar.selectbox('Select Model', ['Decision Tree', 'Linear Regression'])

# Input form
st.header('Employee Information')

# Demographic
col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age', min_value=18, max_value=65, value=30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
with col2:
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    overtime = st.selectbox('Overtime', ['No', 'Yes'])

# Job Information
col1, col2 = st.columns(2)
with col1:
    department = st.selectbox('Department', ['Research & Development', 'Sales', 'Human Resources'])
    job_level = st.number_input('Job Level', min_value=1, max_value=5, value=2)
with col2:
    job_role = st.selectbox('Job Role', [
        'Research Scientist', 'Laboratory Technician', 'Sales Executive',
        'Manufacturing Director', 'Healthcare Representative', 'Manager',
        'Sales Representative', 'Research Director'
    ])
    monthly_income = st.number_input('Monthly Income', min_value=1000, max_value=25000, value=5000)

# Work Environment
col1, col2 = st.columns(2)
with col1:
    business_travel = st.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
    work_life_balance = st.slider('Work Life Balance (1-4)', 1, 4, 3)
with col2:
    environment_satisfaction = st.slider('Environment Satisfaction (1-4)', 1, 4, 3)
    training_times = st.number_input('Training Times Last Year', min_value=0, max_value=10, value=2)

# Create input dictionary (matches your encoding style)
input_data = {
    'Age': age,
    'Gender': 1 if gender == 'Female' else 0,
    'MaritalStatus_Divorced': 1 if marital_status == 'Divorced' else 0,
    'MaritalStatus_Married': 1 if marital_status == 'Married' else 0,
    'MaritalStatus_Single': 1 if marital_status == 'Single' else 0,
    'OverTime': 1 if overtime == 'Yes' else 0,
    'Department_Sales': 1 if department == 'Sales' else 0,
    'Department_Human Resources': 1 if department == 'Human Resources' else 0,
    'JobRole_Laboratory Technician': 1 if job_role == 'Laboratory Technician' else 0,
    'JobRole_Manufacturing Director': 1 if job_role == 'Manufacturing Director' else 0,
    'JobRole_Research Director': 1 if job_role == 'Research Director' else 0,
    'JobRole_Research Scientist': 1 if job_role == 'Research Scientist' else 0,
    'JobRole_Sales Executive': 1 if job_role == 'Sales Executive' else 0,
    'JobRole_Sales Representative': 1 if job_role == 'Sales Representative' else 0,
    'JobLevel': job_level,
    'MonthlyIncome': monthly_income,
    'BusinessTravel_Travel_Frequently': 1 if business_travel == 'Travel_Frequently' else 0,
    'BusinessTravel_Travel_Rarely': 1 if business_travel == 'Travel_Rarely' else 0,
    'WorkLifeBalance': work_life_balance,
    'EnvironmentSatisfaction': environment_satisfaction,
    'TrainingTimesLastYear': training_times,
    # Default values for other features
    'DailyRate': 800,
    'Education': 2,
    'EducationField_Life Sciences': 1,
    'EducationField_Marketing': 0,
    'EducationField_Medical': 0,
    'EducationField_Other': 0,
    'EducationField_Technical Degree': 0,
    'HourlyRate': 60,
    'JobInvolvement': 3,
    'NumCompaniesWorked': 2,
    'PercentSalaryHike': 15,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 3,
    'StockOptionLevel': 1,
    'TotalWorkingYears': 7,
    'YearsAtCompany': 5,
    'YearsInCurrentRole': 4,
    'YearsSinceLastPromotion': 1,
    'YearsWithCurrManager': 3,
    'DistanceFromHome': 10
}

# Convert to DataFrame with correct column order
input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=feature_columns).fillna(0)

# Make prediction
if st.button('Predict Job Satisfaction'):
    if model_type == 'Decision Tree':
        prediction = dt_model.predict(input_df)[0]
    else:
        prediction = lr_model.predict(input_df)[0]
        prediction = max(1, min(4, round(prediction)))  # Ensure 1-4 scale
    
    satisfaction_levels = {
        1: "😞 Very Dissatisfied",
        2: "😕 Somewhat Dissatisfied",
        3: "🙂 Somewhat Satisfied",
        4: "😊 Very Satisfied"
    }
    
    st.subheader('Prediction Result')
    st.metric('Job Satisfaction', 
             f"{prediction} - {satisfaction_levels[prediction]}",
             help="1 (Very Dissatisfied) to 4 (Very Satisfied)")
    
    # Show feature importance
    st.subheader('Key Influencing Factors')
    if model_type == 'Decision Tree':
        importances = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': dt_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(5)
        
        for _, row in importances.iterrows():
            st.progress(row['Importance'] / importances['Importance'].max())
            st.caption(f"{row['Feature']} (importance: {row['Importance']:.3f})")
    else:
        coefficients = pd.DataFrame({
            'Feature': feature_columns,
            'Coefficient': lr_model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False).head(5)
        
        for _, row in coefficients.iterrows():
            direction = "Positive" if row['Coefficient'] > 0 else "Negative"
            st.progress(abs(row['Coefficient']) / coefficients['Coefficient'].abs().max())
            st.caption(f"{row['Feature']} ({direction} impact: {row['Coefficient']:.3f})")