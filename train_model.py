import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle

print("Training started...")

# Load dataset
df = pd.read_csv('final_water_consumption_fraud_dataset.csv')

# Derived features used by the app during prediction
df['Usage_Ratio'] = df['Current_Month_Usage_KL'] / (df['Avg_Monthly_Usage_KL'] + 1e-10)
df['Billing_Per_Usage'] = df['Billing_Amount'] / (df['Current_Month_Usage_KL'] + 1e-10)
df['Deviation_Ratio'] = df['Usage_Deviation'] / (df['Avg_Monthly_Usage_KL'] + 1e-10)

if 'Fraud_Label' not in df.columns:
    raise ValueError("Fraud_Label column is missing from final_water_consumption_fraud_dataset.csv")

features = [
    'Avg_Monthly_Usage_KL',
    'Current_Month_Usage_KL',
    'Billing_Amount',
    'Payment_Delay_Days',
    'Usage_Deviation',
    'Account_Age_Months',
    'Payment_History_Score',
    'Neighborhood_Avg_Usage',
    'Seasonal_Factor',
    'Previous_Fraud_Flags',
    'Usage_Ratio',
    'Billing_Per_Usage',
    'Deviation_Ratio'
]

X = df[features]
y = df['Fraud_Label'].astype(int)

print("Dataset shape:", df.shape)
print("Fraud distribution:")
print(y.value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(
    kernel='rbf',
    C=1,
    gamma='scale',
    probability=True,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Validation accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

joblib.dump(model, 'model_files/best_fraud_detection_model_SVM.pkl')
joblib.dump(scaler, 'model_files/scaler.pkl')

with open('model_files/feature_names.pkl', 'wb') as f:
    pickle.dump(features, f)

print("Model retrained successfully.")