# 💧 Detection of Fraudulent Behaviour in Drinking Water Consumption

A machine learning-based Flask web application that detects potentially fraudulent water consumption behaviour using an **SVM (Support Vector Machine)** classification model.

## 📌 Project Overview

Water distribution agencies can face **Non-Technical Losses (NTL)** due to unauthorized or fraudulent consumption, meter manipulation, abnormal usage patterns, and other irregularities.

Traditional detection methods often depend on manual inspection, which can be time-consuming and costly.

This project uses machine learning to analyse customer water consumption and billing-related information and classify customers as:

- 🚨 **Fraud**
- ✅ **Non-Fraud**

The system provides a web-based interface where users can perform both **single-customer** and **bulk CSV-based predictions**.

---

## 🎯 Objectives

- Detect suspicious water consumption behaviour using machine learning.
- Reduce dependency on manual inspection.
- Identify customers who may require further investigation.
- Provide fraud probability along with prediction results.
- Provide a simple and professional web interface for predictions.

---

## 🧠 Machine Learning Model

The project uses **Support Vector Machine (SVM)** for classification.

### Algorithm

**SVM (Support Vector Machine)**

The SVM model is trained to distinguish between fraudulent and non-fraudulent consumption patterns.

### Data Preprocessing

- Feature engineering
- Train-test splitting
- Feature scaling using `StandardScaler`
- Class balancing using `class_weight='balanced'`

### Engineered Features

The following additional features are calculated:

- `Usage_Ratio`
- `Billing_Per_Usage`
- `Deviation_Ratio`

---

## 📊 Dataset Features

The model uses customer consumption, billing, payment, and account-related information.

| Feature | Description |
|---|---|
| Avg_Monthly_Usage_KL | Average monthly water consumption |
| Current_Month_Usage_KL | Current month's consumption |
| Billing_Amount | Customer billing amount |
| Payment_Delay_Days | Number of delayed payment days |
| Usage_Deviation | Difference from normal usage |
| Account_Age_Months | Age of the customer account |
| Payment_History_Score | Payment behaviour score |
| Neighborhood_Avg_Usage | Average usage in the customer's neighbourhood |
| Seasonal_Factor | Seasonal variation factor |
| Previous_Fraud_Flags | Previous fraud-related flags |
| Usage_Ratio | Current usage compared with average usage |
| Billing_Per_Usage | Billing amount relative to usage |
| Deviation_Ratio | Usage deviation relative to average usage |

---

## 🏗️ System Architecture

```text
Customer Water Consumption Data
              ↓
       Data Preprocessing
              ↓
        Feature Engineering
              ↓
       StandardScaler
              ↓
       SVM Classification
              ↓
       Fraud Probability
              ↓
       Prediction Result
              ↓
        Flask Web Interface