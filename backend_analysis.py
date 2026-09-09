# ================= IMPORT LIBRARIES =================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# ================= LOAD DATASET =================
df = pd.read_csv('final_water_consumption_fraud_dataset.csv')

print("Dataset Shape:", df.shape)
print(df.head())

# ================= DATA PREPROCESSING =================

numerical_cols = [
    'Avg_Monthly_Usage_KL', 'Current_Month_Usage_KL', 'Billing_Amount',
    'Payment_Delay_Days', 'Usage_Deviation', 'Account_Age_Months',
    'Payment_History_Score', 'Neighborhood_Avg_Usage',
    'Seasonal_Factor', 'Previous_Fraud_Flags'
]

# Handle outliers (capping)
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

for col in numerical_cols:
    lower = Q1[col] - 1.5 * IQR[col]
    upper = Q3[col] + 1.5 * IQR[col]
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# ================= FEATURE ENGINEERING =================

df['Usage_Ratio'] = df['Current_Month_Usage_KL'] / (df['Avg_Monthly_Usage_KL'] + 1e-10)
df['Billing_Per_Usage'] = df['Billing_Amount'] / (df['Current_Month_Usage_KL'] + 1e-10)
df['Deviation_Ratio'] = df['Usage_Deviation'] / (df['Avg_Monthly_Usage_KL'] + 1e-10)
 
# ================= FEATURE CORRELATION HEATMAP =================
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=False)

plt.title("Feature Correlation Matrix")
plt.show()

# ================= SPLIT DATA =================

X = df.drop(['Customer_ID', 'Fraud_Label'], axis=1)
y = df['Fraud_Label']

# 🚨 REMOVE DATA LEAKAGE FEATURES
leak_features = ['Fraud_Probability']  # if present in dataset

for col in leak_features:
    if col in X.columns:
        X = X.drop(col, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ================= SCALING =================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= MODEL 1: SVM =================

svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)

print("\nTraining SVM...")
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)
y_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_prob_svm)

print("\nSVM Performance:")
print(f"Accuracy: {accuracy_svm:.4f}")
print(f"Precision: {precision_svm:.4f}")
print(f"Recall: {recall_svm:.4f}")
print(f"F1 Score: {f1_svm:.4f}")
print(f"ROC-AUC: {roc_auc_svm:.4f}")
# ================= SVM CONFUSION MATRIX =================
plt.figure(figsize=(6, 4))
cm_svm = confusion_matrix(y_test, y_pred_svm)

sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'])

plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ================= SVM ROC CURVE =================
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)

plt.figure(figsize=(6, 4))
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={roc_auc_svm:.3f})")
plt.plot([0,1], [0,1], 'k--')

plt.title("SVM ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ================= MODEL 2: KNN =================

knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')

print("\nTraining KNN...")
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
y_prob_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
roc_auc_knn = roc_auc_score(y_test, y_prob_knn)

print("\nKNN Performance:")
print(f"Accuracy: {accuracy_knn:.4f}")
print(f"Precision: {precision_knn:.4f}")
print(f"Recall: {recall_knn:.4f}")
print(f"F1 Score: {f1_knn:.4f}")
print(f"ROC-AUC: {roc_auc_knn:.4f}")

# ================= KNN CONFUSION MATRIX =================
plt.figure(figsize=(6, 4))
cm_knn = confusion_matrix(y_test, y_pred_knn)

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'])

plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ================= MODEL COMPARISON =================

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
    'SVM': [accuracy_svm, precision_svm, recall_svm, f1_svm, roc_auc_svm],
    'KNN': [accuracy_knn, precision_knn, recall_knn, f1_knn, roc_auc_knn]
})

print("\nModel Comparison:")
print(comparison_df)

# ================= MODEL COMPARISON GRAPH =================
comparison_df.set_index('Metric').plot(kind='bar', figsize=(8,5))

plt.title("Model Comparison: SVM vs KNN")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================= SAVE BEST MODEL =================

if f1_svm > f1_knn:
    best_model = svm_model
    model_name = 'SVM'
else:
    best_model = knn_model
    model_name = 'KNN'

# 🔥 SAVE DIRECTLY INTO YOUR PROJECT FOLDER
joblib.dump(best_model, f'model_files/best_fraud_detection_model_{model_name}.pkl')
joblib.dump(scaler, 'model_files/scaler.pkl')

feature_names = list(X.columns)
with open('model_files/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("\n✅ Model Saved Successfully!")

# ================= TEST MODEL =================

model = joblib.load(f'model_files/best_fraud_detection_model_{model_name}.pkl')
scaler = joblib.load('model_files/scaler.pkl')

sample = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)[0]
prob = model.predict_proba(sample_scaled)[0]

print("\nTest Prediction:")
print("Prediction:", "FRAUD" if prediction == 1 else "NON-FRAUD")
print("Probability:", prob)