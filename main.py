# import streamlit as st 
# import pandas as pd 
# import joblib
# from pathlib import Path


# #  LOAD THE PRE-TRAINED MODEL ONLY

# MODEL_PATH = Path(__file__).resolve().parent / 'gb.pkl'


# def load_model():
#     return joblib.load(MODEL_PATH)


# input_columns = [
#     'AccountWeeks',
#     'ContractRenewal',
#     'DataPlan',
#     'DataUsage',
#     'CustServCalls',
#     'DayMins',
#     'DayCalls',
#     'MonthlyCharge',
#     'OverageFee',
#     'RoamMins'
# ]


# #  WEB USER INTERFACE 

# st.title('Customer Churn Predictor')
# st.markdown('Enter customer metrics below to evaluate risk profile and predict retention outcomes.')

# # Create side-by-side input forms for clean layout alignment
# col1, col2 = st.columns(2)

# with col1:
#     account_weeks = st.number_input("Account Age (Weeks active)", min_value=0, step=1, value=100)
#     contract_renewal_text = st.selectbox("Contract Renewed Recently?", ["Yes", "No"])
#     data_plan_text = st.selectbox("Active Data Plan?", ["Yes", "No"])
#     data_usage = st.number_input("Data Usage (GB per month)", min_value=0.0, step=0.1, value=2.7)
#     cust_serv_calls = st.slider("Customer Service Calls Placed", 0, 15, 1)

# with col2:
#     day_mins = st.number_input("Average Day Minutes Used", min_value=0.0, step=1.0, value=200.0)
#     day_calls = st.number_input("Average Day Calls Placed", min_value=0, step=1, value=100)
#     monthly_charge = st.number_input("Monthly Fixed Charge ($)", min_value=0.0, step=1.0, value=60.0)
#     overage_fee = st.number_input("Accrued Overage Fees ($)", min_value=0.0, step=1.0, value=10.0)
#     roam_mins = st.number_input("International Roaming Minutes Used", min_value=0.0, step=1.0, value=10.0)


# # PREDICTION MAPPING & EVALUATION LOGIC

# if st.button("Evaluate Churn Risk", type="primary"):
#     with st.spinner("Loading the prediction model..."):
#         model = load_model()

#         # Convert text selections to matching numeric binary values (0 or 1)
#         contract_renewal = 1 if contract_renewal_text == "Yes" else 0
#         data_plan = 1 if data_plan_text == "Yes" else 0

#         # Build the dictionary mapped to your exact dataset column names
#         new_data = {
#             'AccountWeeks': account_weeks,
#             'ContractRenewal': contract_renewal,
#             'DataPlan': data_plan,
#             'DataUsage': data_usage,
#             'CustServCalls': cust_serv_calls,
#             'DayMins': day_mins,
#             'DayCalls': day_calls,
#             'MonthlyCharge': monthly_charge,
#             'OverageFee': overage_fee,
#             'RoamMins': roam_mins
#         }

#         # Convert to DataFrame matching the structure your model expects
#         df = pd.DataFrame([new_data], columns=input_columns)

#         # Run prediction directly using the raw inputs DataFrame.
#         prediction = model.predict(df)

#         # Support both numeric labels and descriptive labels.
#         result = prediction[0]
#         is_churn = result == 1 or result == "Customer has cancelled service"

#     st.markdown("---")

#     # Display clean outcomes based on target variables (1 = Churn, 0 = Active)
#     if is_churn:
#         st.error('⚠️ Warning: This customer is likely to CHURN.')
#     else:
#         st.success('✅ Safe: This customer is highly likely to STAY.')



import streamlit as st 
import pandas as pd 
import joblib
from pathlib import Path

# ==========================================
# 1. SAFELY CACHE AND LOAD THE PRE-TRAINED MODEL
# ==========================================
# This calculates the folder location safely
MODEL_PATH = Path(__file__).resolve().parent / 'gb.pkl'

@st.cache_resource
def load_model():
    """Loads the model once and keeps it in memory to prevent freezing."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"❌ **File Not Found:** Could not locate your model file at: `{MODEL_PATH}`")
        return None

# Pre-load model on application startup
model = load_model()

# The exact 10 columns your model expects to see, in order
input_columns = [
    'AccountWeeks',
    'ContractRenewal',
    'DataPlan',
    'DataUsage',
    'CustServCalls',
    'DayMins',
    'DayCalls',
    'MonthlyCharge',
    'OverageFee',
    'RoamMins'
]

# ==========================================
# 2. WEB USER INTERFACE (Your Design Layout)
# ==========================================
st.title('Customer Churn Predictor')
st.markdown('Enter customer metrics below to evaluate risk profile and predict retention outcomes.')

# Create side-by-side input forms for clean layout alignment
col1, col2 = st.columns(2)

with col1:
    account_weeks = st.number_input("Account Age (Weeks active)", min_value=0, step=1, value=100)
    contract_renewal_text = st.selectbox("Contract Renewed Recently?", ["Yes", "No"])
    data_plan_text = st.selectbox("Active Data Plan?", ["Yes", "No"])
    data_usage = st.number_input("Data Usage (GB per month)", min_value=0.0, step=0.1, value=2.7)
    cust_serv_calls = st.slider("Customer Service Calls Placed", 0, 15, 1)

with col2:
    day_mins = st.number_input("Average Day Minutes Used", min_value=0.0, step=1.0, value=200.0)
    day_calls = st.number_input("Average Day Calls Placed", min_value=0, step=1, value=100)
    monthly_charge = st.number_input("Monthly Fixed Charge ($)", min_value=0.0, step=1.0, value=60.0)
    overage_fee = st.number_input("Accrued Overage Fees ($)", min_value=0.0, step=1.0, value=10.0)
    roam_mins = st.number_input("International Roaming Minutes Used", min_value=0.0, step=1.0, value=10.0)

# ==========================================
# 3. PREDICTION MAPPING & EVALUATION LOGIC
# ==========================================
if st.button("Evaluate Churn Risk", type="primary"):
    if model is not None:
        with st.spinner("Analyzing profile patterns..."):
            # Convert text selections to matching numeric binary values (0 or 1)
            contract_renewal = 1 if contract_renewal_text == "Yes" else 0
            data_plan = 1 if data_plan_text == "Yes" else 0

            # Build the dictionary mapped to your exact dataset column names
            new_data = {
                'AccountWeeks': account_weeks,
                'ContractRenewal': contract_renewal,
                'DataPlan': data_plan,
                'DataUsage': data_usage,
                'CustServCalls': cust_serv_calls,
                'DayMins': day_mins,
                'DayCalls': day_calls,
                'MonthlyCharge': monthly_charge,
                'OverageFee': overage_fee,
                'RoamMins': roam_mins
            }

            # Convert to DataFrame matching the structure your model expects
            df = pd.DataFrame([new_data], columns=input_columns)

            # Run prediction directly using the raw inputs DataFrame.
            prediction = model.predict(df)

            # Support both numeric labels and descriptive labels.
            result = prediction[0]
            is_churn = result == 1 or str(result).strip().lower() in ["customer has cancelled service", "1"]

        st.markdown("---")

        # Display clean outcomes based on target variables (1 = Churn, 0 = Active)
        if is_churn:
            st.error('⚠️ Warning: This customer is likely to CHURN.')
        else:
            st.success('✅ Safe: This customer is highly likely to STAY.')
    else:
        st.error("Cannot evaluate. The model file failed to load on startup. Check file presence.")
