# Employee Leaving/Attrition Predictor

A machine learning model that predicts whether an employee is likely to leave a company, based on their profile. Built with a Random Forest classifier and deployed via an interactive Streamlit UI.

---

## Overview

Employee attrition is a costly challenge for organizations. This project uses the IBM HR Analytics dataset (1,470 employees, 35 features) to train a classification model that estimates attrition risk and surfaces the probability of an employee staying or leaving, giving HR teams an actionable early-warning signal.

---

## Project Structure

```
├── employee_data.csv          
├── employee_predictor.ipynb   
├── model.pkl                  
├── columns.pkl             
└── app.py                   
```

---

## Notebook

### 1. Exploratory Data Analysis
- Checked for nulls and duplicates
- Visualized attrition distribution and key feature relationships (distance from home, years at company, salary hike, monthly income)

### 2. Feature Engineering
- **Binary encoding:** `Attrition`, `Gender`, `Over18`, `OverTime`
- **One-hot encoding:** `BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`
- **Dropped constants:** `EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`

### 3. Modelling

Two Random Forest models were trained and compared:

| | Model 1 | Model 2 |
|---|---|---|
| Features | All (post-encoding) | Selective (top features by importance) |
| Class imbalance handling | None | SMOTE oversampling |
| Tuning | Default | RandomizedSearchCV (ROC-AUC, 5-fold CV) |

Model 2 (selective features + SMOTE) was selected and saved as `model.pkl`.

---

## Streamlit App

The app provides an interactive form covering all model input features, grouped into:

- **Numeric inputs** – Age, Monthly Income, Daily/Hourly/Monthly Rate
- **Sliders** – Distance From Home, Years at Company, Promotion history, Training times, etc.
- **Dropdowns** – OverTime, Job Involvement, Satisfaction scores, Business Travel, Department, Job Role, Marital Status

On clicking **Predict Attrition**, the app returns:
- A clear risk label (⚠️ High Risk / ✅ Low Risk)
- Stay probability and Leave probability as metrics
- An expandable view of the exact input sent to the model

---

## Tech Stack

| Layer | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn` (Random Forest, train/test split, cross-validation, RandomizedSearchCV) |
| Class imbalance | `imbalanced-learn` (SMOTE) |
| Model serialization | `joblib` |
| Web UI | `streamlit` |

---

## Dataset

IBM HR Analytics Employee Attrition & Performance dataset — 1,470 employee records with 35 features covering demographics, compensation, role, satisfaction scores, and work history.
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
---

## License

This project is for educational and demonstration purposes.
