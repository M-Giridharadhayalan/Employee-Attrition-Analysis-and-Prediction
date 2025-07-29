import streamlit as st
import pickle
import pandas as pd

# Load models and feature columns using pickle
with open('.venv\\Include\\attrition_log.pkl', 'rb') as h:
    logreg = pickle.load(h)

with open('.venv\\Include\\attrition_rfc.pkl', 'rb') as e:
    rf = pickle.load(e)

with open('.venv\\Include\\attrition_feature_columns.pkl', 'rb') as n:
    feature_columns = pickle.load(n)

# Streamlit app
st.title('Employee Attrition Predictor')

# Model selection
model_type = st.sidebar.selectbox('Select Model', ['Logistic Regression', 'Random Forest'])

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
    work_life_balance = st.number_input('Work Life Balance (1-4)', min_value=1, max_value=4, value=3)
with col2:
    environment_satisfaction = st.number_input('Environment Satisfaction (1-4)', min_value=1, max_value=4, value=3)
    job_satisfaction = st.number_input('Job Satisfaction (1-4)', min_value=1, max_value=4, value=3)

# Create input dictionary
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
    'JobSatisfaction': job_satisfaction,
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
    'TrainingTimesLastYear': 2,
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
if st.button('Predict Attrition Risk'):
    if model_type == 'Logistic Regression':
        prediction = logreg.predict(input_df)[0]
        probability = logreg.predict_proba(input_df)[0][1]
    else:
        prediction = rf.predict(input_df)[0]
        probability = rf.predict_proba(input_df)[0][1]
    
    result = "High Risk of Attrition (Likely to Leave)" if prediction == 1 else "Low Risk of Attrition (Likely to Stay)"
    st.subheader('Prediction Result')
    st.write(f'Prediction: **{result}**')
    st.write(f'Probability: **{probability:.2%}**')
    
    # Show important features
    st.subheader('Key Factors Influencing Prediction')
    if model_type == 'Logistic Regression':
        coef_df = pd.DataFrame({
            'Feature': feature_columns,
            'Coefficient': logreg.coef_[0]
        }).sort_values('Coefficient', ascending=False)
        
        top_factors = coef_df.head(3) if prediction == 1 else coef_df.tail(3)
        
        for _, row in top_factors.iterrows():
            effect = "Increasing risk" if row['Coefficient'] > 0 else "Decreasing risk"
            st.write(f"- {row['Feature']}: {effect}")
    else:
        importances = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False).head(3)
        
        for _, row in importances.iterrows():
            st.write(f"- {row['Feature']}: Importance {row['Importance']:.3f}")