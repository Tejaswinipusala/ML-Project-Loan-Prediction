# 🏦 Loan Prediction System using Machine Learning & FastAPI

## 📌 Project Overview

The Loan Prediction System is an end-to-end Machine Learning project that predicts whether a loan application will be **Approved** or **Rejected** based on an applicant's financial and personal information.

The project includes:

* Data preprocessing
* Feature engineering
* Machine Learning model training
* Model evaluation
* Model serialization using Pickle
* FastAPI REST API
* Interactive Swagger UI for testing
* GitHub version control

---

# 🎯 Problem Statement

Financial institutions receive thousands of loan applications every day. Evaluating each application manually is time-consuming and may introduce inconsistencies.

This project automates the loan approval prediction process using Machine Learning so that decisions can be made quickly and consistently.

---

# 🚀 Features

* Loan Approval Prediction
* Random Forest Classifier
* Data Cleaning
* Date Feature Extraction (Year, Month, Day)
* One-Hot Encoding of Categorical Features
* Model Evaluation
* Pickle Model Serialization
* REST API using FastAPI
* Interactive Swagger UI
* JSON Input & Output
* Probability Scores for Approval/Rejection

---

# 📂 Dataset Information

The dataset contains **20,000 loan application records** with applicant financial details.

### Features Used

* Age
* AnnualIncome
* CreditScore
* EmploymentStatus
* EducationLevel
* Experience
* LoanAmount
* LoanDuration
* MaritalStatus
* NumberOfDependents
* HomeOwnershipStatus
* MonthlyDebtPayments
* CreditCardUtilizationRate
* NumberOfOpenCreditLines
* NumberOfCreditInquiries
* DebtToIncomeRatio
* BankruptcyHistory
* LoanPurpose
* PreviousLoanDefaults
* PaymentHistory
* LengthOfCreditHistory
* SavingsAccountBalance
* CheckingAccountBalance
* TotalAssets
* TotalLiabilities
* MonthlyIncome
* UtilityBillsPaymentHistory
* JobTenure
* NetWorth
* BaseInterestRate
* InterestRate
* MonthlyLoanPayment
* TotalDebtToIncomeRatio
* RiskScore
* Year
* Month
* Day

### Target Variable

LoanApproved

* 1 → Loan Approved
* 0 → Loan Rejected

---

# ⚙️ Data Preprocessing

The following preprocessing steps were performed:

* Removed unnecessary columns
* Converted ApplicationDate into:

  * Year
  * Month
  * Day
* One-Hot Encoding for categorical variables
* Train-Test Split
* Feature Scaling (experimented using StandardScaler and MinMaxScaler)
* Model Training

---

# 🤖 Machine Learning Model

**Algorithm Used**
Logistic Regression Classifier
Random Forest Classifier

Reason for choosing Random Forest:

* Handles mixed data types
* Works well on tabular datasets
* Reduces overfitting compared to a single decision tree
* Provides prediction probabilities

---

# 📊 Model Evaluation

Evaluation metrics used:

* Accuracy Score
* Classification Report
* Confusion Matrix
* ROC-AUC Score

---

# 🛠 Tech Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Pydantic
* Pickle

### Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub
* Swagger UI

---

# 📁 Project Structure

```
ML-Project-Loan-Prediction/
│
├── app.py
├── Loan Prediction CP.ipynb
├── Loan.csv
├── model.pkl
├── model_columns.pkl
├── requirements.txt
├── README.md
└── images/
```

---

# 🌐 FastAPI Endpoint

### Predict Loan

```
POST /predict
```

---

# 📥 Sample Request

```json
{
  "Age": 30,
  "AnnualIncome": 85000,
  "CreditScore": 780,
  "EmploymentStatus": "Employed",
  "EducationLevel": "Master",
  "Experience": 8,
  "LoanAmount": 150000,
  "LoanDuration": 60,
  "MaritalStatus": "Married",
  "NumberOfDependents": 2,
  "HomeOwnershipStatus": "Own",
  "MonthlyDebtPayments": 1200,
  "CreditCardUtilizationRate": 0.25,
  "NumberOfOpenCreditLines": 4,
  "NumberOfCreditInquiries": 1,
  "DebtToIncomeRatio": 0.20,
  "BankruptcyHistory": 0,
  "LoanPurpose": "Home",
  "PreviousLoanDefaults": 0,
  "PaymentHistory": 98,
  "LengthOfCreditHistory": 12,
  "SavingsAccountBalance": 40000,
  "CheckingAccountBalance": 12000,
  "TotalAssets": 550000,
  "TotalLiabilities": 120000,
  "MonthlyIncome": 7000,
  "UtilityBillsPaymentHistory": 0.98,
  "JobTenure": 8,
  "NetWorth": 430000,
  "BaseInterestRate": 0.06,
  "InterestRate": 0.075,
  "MonthlyLoanPayment": 2600,
  "TotalDebtToIncomeRatio": 0.23,
  "RiskScore": 28,
  "Year": 2023,
  "Month": 5,
  "Day": 12
}
```

---

# 📤 Sample Response

```json
{
  "prediction": 1,
  "income_class": "Loan Approved",
  "probability_Loan Rejected": 0.08,
  "probability_Loan Approved": 0.92
}
```

---

# ▶️ How to Run the Project

### Clone Repository

```
git clone https://github.com/Tejaswinipusala/ML-Project-Loan-Prediction.git
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run FastAPI

```
uvicorn app:app --reload
```

### Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

# 📸 Swagger UI

screenshots of:

* Swagger Home
* Sample Prediction
* Successful API Response

Store images inside:

```
Home Page of Swagger UI
<img width="1918" height="1077" alt="image" src="https://github.com/user-attachments/assets/a9d44cdb-4b65-4ddf-a300-5a761cd43d3f" />
```
#Predict
<img width="1918" height="1077" alt="Screenshot 2026-06-30 101736" src="https://github.com/user-attachments/assets/2f5c7914-7f41-4c1c-b0c4-4cc4a31d0d97" />

#Sample input
<img width="1765" height="907" alt="Screenshot 2026-06-30 101756" src="https://github.com/user-attachments/assets/ae9b7f37-4524-4749-a117-3ad96a700943" />

#Result(Loan approved or rejected based on input)--Loan Approved✅
<img width="1918" height="1078" alt="Screenshot 2026-06-30 101811" src="https://github.com/user-attachments/assets/c877098c-6f72-4187-8c7b-f48b8f92f666" />
<img width="1918" height="1077" alt="Screenshot 2026-06-30 101818" src="https://github.com/user-attachments/assets/c9a5e2ed-b243-4dd9-8441-735e3fac9ab4" />
---

# 🔮 Future Improvements

* Hyperparameter tuning
* Model comparison using XGBoost and LightGBM
* Explainable AI using SHAP
* Docker containerization
* CI/CD pipeline using GitHub Actions
* Cloud deployment (Render/AWS/Azure)
* User-friendly web interface with Streamlit or React
* Database integration for storing predictions

---

# 👩‍💻 Author

**Tejaswini Pusala**

Aspiring AI/ML & Data Science Engineer

GitHub: https://github.com/Tejaswinipusala

