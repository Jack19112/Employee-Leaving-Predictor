import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👤", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

model, columns = load_model()

st.title("Employee Attrition Predictor")
st.caption("Predicts whether an employee is likely to leave based on their profile.")

st.subheader("Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=35)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000, step=500)
    daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
    hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65)
    monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=27000, value=14000, step=500)

with col2:
    distance_from_home = st.slider("Distance From Home (km)", 1, 29, 5)
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    total_working_years = st.slider("Total Working Years", 0, 40, 10)
    years_with_curr_manager = st.slider("Years With Current Manager", 0, 17, 3)

with col3:
    num_companies_worked = st.slider("Num Companies Worked", 0, 9, 2)
    percent_salary_hike = st.slider("Percent Salary Hike", 11, 25, 15)
    training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)
    years_in_current_role = st.slider("Years in Current Role", 0, 18, 3)
    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])

st.divider()
col4, col5 = st.columns(2)

with col4:
    over_time = st.selectbox("OverTime", ["Yes", "No"])
    job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)

with col5:
    environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
    relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2)
    work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], index=2)
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5], index=2)

st.divider()
col6, col7 = st.columns(2)

with col6:
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])

with col7:
    job_role = st.selectbox("Job Role", [
        "Healthcare Representative", "Human Resources", "Laboratory Technician",
        "Manager", "Manufacturing Director", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative"
    ])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])




def build_input():
    row = {
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EnvironmentSatisfaction": environment_satisfaction,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": 1 if over_time == "Yes" else 0,
        "PercentSalaryHike": percent_salary_hike,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    }



    for col in columns:
        if col not in row and col != "Attrition":
            row[col] = 0

    if business_travel in row:
        row[business_travel] = 1

    dept_col = f"Department_{department}"
    if dept_col in row:
        row[dept_col] = 1

    role_col = f"JobRole_{job_role}"
    if role_col in row:
        row[role_col] = 1

    marital_col = f"MaritalStatus_{marital_status}"
    if marital_col in row:
        row[marital_col] = 1

    model_cols = [c for c in columns if c != "Attrition"]
    df = pd.DataFrame([row])
    df = df.reindex(columns=model_cols, fill_value=0)
    return df




if st.button("Predict Attrition", type="primary", use_container_width=True):
    input_df = build_input()
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.divider()
    if prediction == 1:
        st.error("⚠️ High Attrition Risk", icon="🚨")
    else:
        st.success("✅ Low Attrition Risk", icon="✅")

    col_a, col_b = st.columns(2)
    col_a.metric("Stay probability", f"{proba[0]*100:.1f}%")
    col_b.metric("Leave probability", f"{proba[1]*100:.1f}%")

    with st.expander("Show input sent to model"):
        st.dataframe(build_input())