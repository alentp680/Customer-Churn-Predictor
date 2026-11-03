#  Customer Churn Predictor

A Machine Learning web application that predicts whether a telecom customer is likely to churn based on their demographic, service, contract, and billing information.

The project uses the IBM Telco Customer Churn dataset and provides an interactive prediction interface built with Streamlit.

---

##  Features

- Predict whether a customer is likely to churn
- Display churn and retention probabilities
- Interactive customer input form
- Machine Learning pipeline with preprocessing
- One-Hot Encoding for categorical features
- StandardScaler for numerical features
- Logistic Regression classification model
- Prediction timestamp
- Suggested actions for customers likely to churn
- Customer information summary

---

##  Machine Learning

The final deployed model is:

**Logistic Regression**

The model is implemented using a Scikit-learn Pipeline.

### Preprocessing

Numerical features:

- Tenure Months
- Monthly Charges
- Total Charges

Numerical preprocessing:

- StandardScaler

Categorical features are processed using:

- One-Hot Encoding
- `handle_unknown="ignore"`

---

##  Dataset

The project uses the **IBM Telco Customer Churn dataset**.

The dataset contains information about telecom customers, including:

- Customer demographics
- Services
- Contract information
- Payment method
- Monthly charges
- Total charges
- Churn status

Dataset size:

- **7,043 customers**
- **19 input features used for prediction**

---

##  Model Evaluation

Several classification models were evaluated during development:

| Model | Accuracy |
|---|---:|
| Logistic Regression | ~80.53% |
| Decision Tree | ~72.92% |
| Random Forest | ~79.25% |

Logistic Regression was selected as the final deployment model because it provided the best overall performance among the tested models.

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- SHAP
- Jupyter Notebook
- Git
- GitHub

---

##  Project Structure

```text
Customer Churn Predictor
│
├── app/
│   └── app.py
│
├── models/
│   ├── churn_model.pkl
│   └── feature_names.pkl
│
├── data/
│   └── raw/
│       ├── Telco_customer_churn.csv
│       └── Telco_customer_churn.xlsx
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── utils/
│   ├── predictor.py
│   └── explain.py
│
├── .gitignore
├── requirements.txt
└── README.md