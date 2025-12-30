import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained pipeline
model = joblib.load('models/xgb_fraud_pipeline.pkl')

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection")

# -------------------
# Layout: Left inputs, Right outputs
# -------------------
left_col, right_col = st.columns([2, 1])  

# -------------------
# Inputs in left column
# -------------------
with left_col:
    st.subheader("Transaction Details")
    
    amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0, key='amt')
    hour = st.slider("Transaction Hour", 0, 23, 12, key='hour')
    day = st.slider("Transaction Day", 1, 31, 1, key='day')
    month = st.slider("Transaction Month", 1, 12, 1, key='month')
    age = st.number_input("Customer Age", min_value=0, max_value=120, value=30, key='age')
    
    gender_display = st.selectbox("Gender", ["Male", "Female"], key='gender')
    category_display = st.selectbox(
        "Category",
        [
            "Food & Dining", "Grocery (POS)", "Grocery (Net)", "Miscellaneous (POS)", "Miscellaneous (Net)",
            "Shopping (POS)", "Shopping (Net)", "Gas & Transport", "Travel",
            "Personal Care", "Health & Fitness", "Home", "Kids & Pets"
        ],
        key='category'
    )
    state_display = st.selectbox(
        "State",
        [
            "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","District of Columbia",
            "Florida","Georgia","Hawaii","Iowa","Idaho","Illinois","Indiana","Kansas","Kentucky","Louisiana",
            "Massachusetts","Maryland","Michigan","Minnesota","Missouri","Mississippi","Montana","North Carolina",
            "North Dakota","Nebraska","New Jersey","New Mexico","Nevada","New York","Ohio","Oklahoma","Oregon",
            "Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Virginia",
            "Vermont","Washington","Wisconsin","West Virginia","Wyoming"
        ],
        key='state'
    )

# -------------------
# Map display values back to model keys
# -------------------
category_map = {
    "Food & Dining":"food_dining",
    "Grocery (POS)":"grocery_pos",
    "Grocery (Net)":"grocery_net",
    "Miscellaneous (POS)":"misc_pos",
    "Miscellaneous (Net)":"misc_net",
    "Shopping (POS)":"shopping_pos",
    "Shopping (Net)":"shopping_net",
    "Gas & Transport":"gas_transport",
    "Travel":"travel",
    "Personal Care":"personal_care",
    "Health & Fitness":"health_fitness",
    "Home":"home",
    "Kids & Pets":"kids_pets"
}

state_map = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT",
    "District of Columbia":"DC","Florida":"FL","Georgia":"GA","Hawaii":"HI","Iowa":"IA","Idaho":"ID","Illinois":"IL",
    "Indiana":"IN","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Massachusetts":"MA","Maryland":"MD","Michigan":"MI",
    "Minnesota":"MN","Missouri":"MO","Mississippi":"MS","Montana":"MT","North Carolina":"NC","North Dakota":"ND",
    "Nebraska":"NE","New Jersey":"NJ","New Mexico":"NM","Nevada":"NV","New York":"NY","Ohio":"OH","Oklahoma":"OK",
    "Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN",
    "Texas":"TX","Utah":"UT","Virginia":"VA","Vermont":"VT","Washington":"WA","Wisconsin":"WI","West Virginia":"WV",
    "Wyoming":"WY"
}

category = category_map[category_display]
state = state_map[state_display]
gender = "M" if gender_display=="Male" else "F"

# -------------------
# Preprocessing
# -------------------
input_df = pd.DataFrame({
    'amt':[amt],
    'transaction_hour':[hour],
    'transaction_day':[day],
    'transaction_month':[month],
    'customer_age':[age],
    'is_night':[1 if hour in [22,23,0,1,2,3,4,5] else 0],
    'is_high_amount':[1 if amt>500 else 0],
    'log_amt':[np.log1p(amt)],
    'amt_to_mean':[amt/200],
    'city_pop':[1000],
    'category':[category],
    'gender':[gender],
    'state':[state]
})

# -------------------
# Prediction button
# -------------------
threshold = 0.5  # Fraud probability threshold

if st.button("Predict"):
    try:
        pred_prob = model.predict_proba(input_df)[:,1][0]
        pred_label = 1 if pred_prob >= threshold else 0
        pred_percent = int(pred_prob * 100)
        
        with right_col:
            st.subheader("Prediction Result")
            if pred_label == 1:
                st.error(f"⚠️ Predicted: **FRAUD**")
            else:
                st.success(f"✅ Predicted: **LEGITIMATE**")

            st.info(f"Fraud Probability: **{pred_percent}%**")
            st.progress(float(pred_prob))

    except Exception as e:
        with right_col:
            st.error(f"Error: {e}")