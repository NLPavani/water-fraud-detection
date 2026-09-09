import joblib
import pickle
import numpy as np
import pandas as pd

# Load model
model = joblib.load("model_files/best_fraud_detection_model_SVM.pkl")

# Load scaler
scaler = joblib.load("model_files/scaler.pkl")

# Load feature names
with open("model_files/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

print("Model Type:", type(model).__name__)
print("Scaler Type:", type(scaler).__name__)
print("Feature Names:", feature_names)

# --------------------------------------------------
# Test customer - original 10 input values
# --------------------------------------------------

avg_usage = 100
current_usage = 105
billing_amount = 300
payment_delay = 1
usage_deviation = 5
account_age = 24
payment_history = 90
neighborhood_avg = 110
seasonal_factor = 1
previous_fraud_flags = 0

# --------------------------------------------------
# Create the SAME 3 derived features used in training
# --------------------------------------------------

usage_ratio = current_usage / (avg_usage + 1e-10)

billing_per_usage = billing_amount / (current_usage + 1e-10)

deviation_ratio = usage_deviation / (avg_usage + 1e-10)

# --------------------------------------------------
# Create all 13 features in the SAME ORDER
# --------------------------------------------------

test_customer = pd.DataFrame([[
    avg_usage,
    current_usage,
    billing_amount,
    payment_delay,
    usage_deviation,
    account_age,
    payment_history,
    neighborhood_avg,
    seasonal_factor,
    previous_fraud_flags,
    usage_ratio,
    billing_per_usage,
    deviation_ratio
]], columns=feature_names)

print("\nNumber of input features:", test_customer.shape[1])

# Scale
scaled_customer = scaler.transform(test_customer)

# Prediction
prediction = model.predict(scaled_customer)

# Probability
probability = model.predict_proba(scaled_customer)

print("Prediction:", prediction)
print("Fraud Probability:", probability[0][1])
print("Non-Fraud Probability:", probability[0][0])