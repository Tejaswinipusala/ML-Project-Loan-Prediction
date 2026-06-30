from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

app = FastAPI(title="Loan Prediction API")


class LoanApplication(BaseModel):

    Age: int
    AnnualIncome: float
    CreditScore: int
    EmploymentStatus: str
    EducationLevel: str
    Experience: int
    LoanAmount: float
    LoanDuration: int
    MaritalStatus: str
    NumberOfDependents: int
    HomeOwnershipStatus: str
    MonthlyDebtPayments: float
    CreditCardUtilizationRate: float
    NumberOfOpenCreditLines: int
    NumberOfCreditInquiries: int
    DebtToIncomeRatio: float
    BankruptcyHistory: int
    LoanPurpose: str
    PreviousLoanDefaults: int
    PaymentHistory: int
    LengthOfCreditHistory: int
    SavingsAccountBalance: float
    CheckingAccountBalance: float
    TotalAssets: float
    TotalLiabilities: float
    MonthlyIncome: float
    UtilityBillsPaymentHistory: float
    JobTenure: int
    NetWorth: float
    BaseInterestRate: float
    InterestRate: float
    MonthlyLoanPayment: float
    TotalDebtToIncomeRatio: float
    RiskScore: float
    Year: int
    Month: int
    Day: int


@app.post("/predict")
def predict(data: LoanApplication):

    # Convert request to DataFrame
    df = pd.DataFrame([data.model_dump()])

    # One-hot encode categorical columns
    df = pd.get_dummies(
        df,
        columns=[
            "EmploymentStatus",
            "EducationLevel",
            "MaritalStatus",
            "HomeOwnershipStatus",
            "LoanPurpose"
        ],
        drop_first=True
    )

    # Match training columns
    df = df.reindex(columns=model_columns, fill_value=0)

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    return {
        "prediction": int(pred),
        "loan_status": "Loan Approved" if pred == 1 else "Loan Rejected",
        "approval_probability": round(float(proba[1]), 3),
        "rejection_probability": round(float(proba[0]), 3)
    }