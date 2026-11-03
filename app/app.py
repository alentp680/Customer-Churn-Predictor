import streamlit as st
import pandas as pd

from utils.predictor import predict_churn

from datetime import datetime

# --------------------------
# Load Model
# --------------------------


# --------------------------
# App Title
# --------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📞",
    layout="wide"
)

st.title("📞 Customer Churn Predictor")

st.markdown(
    """
Enter the customer's information in the sidebar and click **Predict Churn**.
"""
)

# --------------------------
# Customer Information
# --------------------------

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    ["Yes", "No"]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure Months",
    min_value=0,
    max_value=72,
    value=12
)


monthly = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)
phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

input_data = pd.DataFrame({
    "Gender": [gender],
    "Senior Citizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "Tenure Months": [tenure],
    "Phone Service": [phone],
    "Multiple Lines": [multiple],
    "Internet Service": [internet],
    "Online Security": [security],
    "Online Backup": [backup],
    "Device Protection": [protection],
    "Tech Support": [support],
    "Streaming TV": [tv],
    "Streaming Movies": [movies],
    "Contract": [contract],
    "Paperless Billing": [paperless],
    "Payment Method": [payment],
    "Monthly Charges": [monthly],
    "Total Charges": [total]
})



if st.button("🔍 Predict Churn", type="primary"):

    with st.spinner("Predicting..."):

        prediction, probability = predict_churn(input_data)

    
    

    confidence = max(probability[0]) * 100

    st.progress(confidence/100)

    st.info(f"Model Confidence: {confidence:.2f}%")
    
    st.subheader("Prediction Result")    

    st.caption(
    f"Prediction generated on {datetime.now():%d %B %Y %H:%M}"
    )
    
    if prediction[0] == 1:

        st.error(
            f"⚠️ Customer is likely to churn ({probability[0][1]:.1%})"
        )

        st.warning("Suggested Action")

        st.write("""
        • Offer a long-term contract
        • Provide a discount
        • Offer free Tech Support
        • Contact the customer before renewal
        """)
    else:
        st.success("✅ Customer is NOT likely to Churn")
        st.balloons()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Stay Probability",
            f"{probability[0][0]:.2%}"
        )
        

    with col2:
        st.metric(
            "Churn Probability",
            f"{probability[0][1]:.2%}"
        )
        
    left, right = st.columns([2,1])     

    with left:
        st.subheader("📋 Customer Details")
        st.dataframe(input_data, use_container_width=True)

    with right:
        st.subheader("📈 Prediction Summary")


    

st.markdown("---")

st.markdown("""
### About

This application predicts whether a telecom customer is likely to churn using a Machine Learning model trained on the IBM Telco Customer Churn dataset.

Model:
- Logistic Regression
- Scikit-learn Pipeline
- One-Hot Encoding
- StandardScaler

Built using:
- Python
- Streamlit
- Scikit-learn
""")

st.sidebar.info(
    "Model: Logistic Regression"
)

