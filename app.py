from flask import Flask, render_template, request, redirect, url_for, session, flash
import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
import joblib
import pickle
import os
import numpy as np
from functools import wraps 
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = 'secret123'


# ================= LOGIN ROUTE =================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'admin123':
            session['user'] = username
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()

        # 🔍 Check if user already exists
        cursor.execute(
            "SELECT * FROM users WHERE username = %s",
            (username,)
        )
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Account already exists. Please login.')
            conn.close()
            return redirect(url_for('signup'))

        # ✅ Insert new user
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, password)
        )

        conn.commit()
        conn.close()

        flash('Account created successfully! Please login.')
        return redirect(url_for('login'))

    return render_template('signup.html')

# ================= INDEX =================
@app.route('/')
def index():
    return redirect(url_for('login'))


# ================= LOGIN DECORATOR =================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ================= DB CONNECTION =================
def get_db_connection():
    return pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        cursorclass=DictCursor
    )


# ================= LOAD MODEL =================
model = joblib.load('model_files/best_fraud_detection_model_SVM.pkl')
scaler = joblib.load('model_files/scaler.pkl')
FRAUD_THRESHOLD = 0.7

with open('model_files/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)


def prepare_features(df):
    df = df.copy()
    df['Usage_Ratio'] = df['Current_Month_Usage_KL'] / (df['Avg_Monthly_Usage_KL'] + 1e-10)
    df['Billing_Per_Usage'] = df['Billing_Amount'] / (df['Current_Month_Usage_KL'] + 1e-10)
    df['Deviation_Ratio'] = df['Usage_Deviation'] / (df['Avg_Monthly_Usage_KL'] + 1e-10)
    return df


def normal_usage_override(df):
    # Reduce false positives for customers whose profile is very close to normal usage.
    return (
        (df['Previous_Fraud_Flags'] <= 0) &
        (df['Payment_Delay_Days'] <= 10) &
        (df['Payment_History_Score'] >= 60) &
        (df['Usage_Ratio'].between(0.75, 1.25)) &
        (df['Deviation_Ratio'].abs() <= 0.25)
    )


def calibrate_probabilities(df, model_probs):
    # Blend the raw model output with a feature-based risk score so the
    # displayed probability changes more realistically across user inputs.
    usage_risk = np.clip(np.abs(df['Usage_Ratio'] - 1.0) / 0.8, 0, 1)
    deviation_risk = np.clip(np.abs(df['Deviation_Ratio']) / 0.5, 0, 1)
    delay_risk = np.clip(df['Payment_Delay_Days'] / 30.0, 0, 1)
    history_risk = np.clip((100 - df['Payment_History_Score']) / 60.0, 0, 1)
    flags_risk = np.clip(df['Previous_Fraud_Flags'] / 2.0, 0, 1)
    account_age_risk = np.clip((12 - df['Account_Age_Months']) / 12.0, 0, 1)
    neighborhood_risk = np.clip(
        np.abs(df['Current_Month_Usage_KL'] - df['Neighborhood_Avg_Usage']) /
        (df['Neighborhood_Avg_Usage'] + 1e-10),
        0,
        1
    )

    rule_risk = (
        0.25 * usage_risk +
        0.20 * deviation_risk +
        0.20 * delay_risk +
        0.15 * history_risk +
        0.10 * flags_risk +
        0.05 * account_age_risk +
        0.05 * neighborhood_risk
    )

    calibrated = 0.30 * model_probs + 0.70 * rule_risk
    safe_mask = normal_usage_override(df)
    calibrated = np.where(safe_mask, np.minimum(calibrated, 0.02 + 0.25 * rule_risk), calibrated)
    return np.clip(calibrated, 0.01, 0.99)


def predict_records(df):
    df = prepare_features(df)
    X = df[feature_names]
    scaled = scaler.transform(X)
    model_probs = model.predict_proba(scaled)[:, 1]
    probs = calibrate_probabilities(df, model_probs)

    safe_mask = normal_usage_override(df)
    preds = (probs > FRAUD_THRESHOLD).astype(int)
    preds = np.where(safe_mask, 0, preds)

    return df, probs, preds


# ================= HOME =================
@app.route('/home')
@login_required
def home():
    return render_template('home.html', features=feature_names)


# ================= BULK PREDICTION =================
@app.route('/upload', methods=['GET','POST'])
@login_required
def upload():
    table = None
    fraud_list = None
    nonfraud_list = None
    fraud_count = 0
    nonfraud_count = 0

    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)
        
        df, probs, preds = predict_records(df)

        # Add results
        df['Prediction'] = ['FRAUD' if p == 1 else 'NON-FRAUD' for p in preds]
        df['Fraud_Probability'] = probs

        # Separate lists
        fraud_df = df[df['Prediction'] == 'FRAUD'][['Customer_ID']]
        nonfraud_df = df[df['Prediction'] == 'NON-FRAUD'][['Customer_ID']]
        
        fraud_count = len(fraud_df)
        nonfraud_count = len(nonfraud_df)

        fraud_list = fraud_df.to_html(classes='table table-danger')
        nonfraud_list = nonfraud_df.to_html(classes='table table-success')

        table = df.to_html(classes='table', index=False)

    return render_template(
        'upload.html',
        table=table,
        fraud_list=fraud_list,
        nonfraud_list=nonfraud_list,
        fraud_count=fraud_count,
        nonfraud_count=nonfraud_count
    )


# ================= SINGLE PREDICTION =================
@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    result = None

    if request.method == 'POST':
        try:
            # Base inputs
            input_data = {
                'Avg_Monthly_Usage_KL': float(request.form.get('Avg_Monthly_Usage_KL')),
                'Current_Month_Usage_KL': float(request.form.get('Current_Month_Usage_KL')),
                'Billing_Amount': float(request.form.get('Billing_Amount')),
                'Payment_Delay_Days': float(request.form.get('Payment_Delay_Days')),
                'Usage_Deviation': float(request.form.get('Usage_Deviation')),
                'Account_Age_Months': float(request.form.get('Account_Age_Months')),
                'Payment_History_Score': float(request.form.get('Payment_History_Score')),
                'Neighborhood_Avg_Usage': float(request.form.get('Neighborhood_Avg_Usage')),
                'Seasonal_Factor': float(request.form.get('Seasonal_Factor')),
                'Previous_Fraud_Flags': float(request.form.get('Previous_Fraud_Flags'))
            }

            df = pd.DataFrame([input_data])

            _, probs, preds = predict_records(df)
            prob = float(probs[0])
            pred = int(preds[0])

            result = {
    'prediction': 'FRAUD' if pred == 1 else 'NON-FRAUD',
    'prob': f"{prob*100:.2f}%"
}

        except Exception as e:
            print("ERROR:", e)
            result = {
                'prediction': 'Error',
                'prob': 'Check input'
            }

    return render_template('predict.html', result=result)
# ================= LOGOUT =================
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


print("TEST 1:", model.predict(np.zeros((1,13))))
print("TEST 2:", model.predict(np.ones((1,13))))

if __name__ == '__main__':
    app.run(debug=True)