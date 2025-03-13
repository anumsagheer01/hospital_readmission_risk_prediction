# 🚑 Patient Readmission Prediction  

## Project Overview  
This project predicts **hospital readmission** for diabetic patients within **30 days** using **machine learning**. Hospitals can use this model to **reduce unnecessary readmissions**, improve patient care, and optimize resource allocation.  

- **Dataset:** 100,000+ patient records from **130 US hospitals** (Kaggle)  
- **Models Used:** **Logistic Regression, Random Forest**  
- **Best Model Accuracy:** **89.27% (Random Forest)**  
- **Deployment:** **Flask API + Docker** for real-time predictions  

---

## Data Preprocessing & Feature Engineering  
**Missing Values Handled:** Imputed missing data for **diagnosis codes & lab results**  
**Categorical Encoding:** Used **OneHotEncoding** for diagnosis categories  
**Feature Selection:** Identified **top 15 risk factors** using **SHAP explainability**  
**Balancing Classes:** Used **SMOTE** to handle **imbalanced data (88% non-readmitted, 12% readmitted)**  

---

## Model Training & Performance  

| Model                  | Accuracy | Precision | Recall | AUC-ROC  |  
|------------------------|----------|-----------|--------|-----------|  
| Logistic Regression    | 85.12%   | 81.3%     | 78.5%  | 0.86      |  
| Random Forest (Best)   | **89.27%**   | **85.6%**     | **82.4%**  | **0.91**      |  

**Hyperparameter tuning (GridSearchCV) improved accuracy from 86.3% → 89.27%**  
**SHAP analysis identified top risk factors (age, insulin levels, prior admissions)**  

---

## Model Deployment (Flask + Docker)  

**Built a Flask API** for real-time predictions  
**Containerized with Docker** for scalability & easy deployment  
**Tested API using Postman & Python requests**  

### 🏗Run the API Locally  
```bash
python app.py

### Send Data to API and get prediction
Once the Flask API is running, you can send patient data via a POST request using Python.

import requests  

# Sample patient data  
data = {  
    "age": 65,  
    "insulin": 2,  
    "time_in_hospital": 5,  
    "num_medications": 8,  
    "A1Cresult": "None"  
}  

# API endpoint  
url = "http://localhost:5000/predict"  

# Send POST request  
response = requests.post(url, json=data)  

# Print response  
print(response.json())  

Sample output:
{"readmission_prediction": 1} 

The API returns 1 or 0, where:

1 → Patient is likely to be readmitted within 30 days
0 → Patient is not likely to be readmitted

This project is built using the following technologies:
Python (Pandas, NumPy, Scikit-learn)
SHAP (for explainability)
Random Forest, Logistic Regression
Flask API for real-time predictions
Docker for containerization
Postman for API testing


