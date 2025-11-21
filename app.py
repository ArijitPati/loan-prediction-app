from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# ================================
#  LOAD MODEL SAFELY
# ================================
MODEL_PATH = r"D:\PROJECT\loan-prediction-app\credit_risk_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully")
else:
    print("❌ Model file NOT found!")
    model = None


#  HOME PAGE

@app.route('/')
def home():
    return render_template('index.html', prediction="", form_data={})


# ================================
#  PREDICTION ROUTE
# ================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form.to_dict()

        # Log inputs
        print("\n--- FORM INPUTS ---")
        print(form)

        # Safe conversion helper
        def safe_cast(value, cast_type, default=0):
            try:
                return cast_type(value)
            except:
                return default

        # Convert inputs safely
        gender = safe_cast(form.get('gender'), int)
        married = safe_cast(form.get('married'), int)
        dependents = safe_cast(form.get('dependents'), int)
        education = safe_cast(form.get('education'), int)
        self_employed = safe_cast(form.get('self_employed'), int)
        applicant_income = safe_cast(form.get('applicant_income'), float)
        coapplicant_income = safe_cast(form.get('coapplicant_income'), float)
        loan_amount = safe_cast(form.get('loan_amount'), float)
        loan_amount_term = safe_cast(form.get('loan_amount_term'), float)
        credit_history = safe_cast(form.get('credit_history'), int)
        property_area = safe_cast(form.get('property_area'), int)

        # Derived features
        total_income = applicant_income + coapplicant_income
        debt_income_ratio = loan_amount / (total_income + 1)

        # Arrange features exactly as TRAINING order
        features = np.array([[
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area,
            total_income, debt_income_ratio
        ]])

        print("\n--- FINAL FEATURES SENT TO MODEL ---")
        print(features)

        # Check model
        if model is None:
            return render_template(
                'index.html',
                prediction="⚠️ Model not loaded.",
                form_data=form
            )

        # Prediction
        pred = model.predict(features)[0]

        # Probability (if available)
        proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None

        # Normalize label types
        label = str(pred).upper()

        if label in ["Y", "1", "APPROVED"]:
            confidence = f" (Confidence: {round(proba[1] * 100, 2)}%)" if proba is not None else ""
            result = f"✅ Loan Approved{confidence}"
        else:
            confidence = f" (Confidence: {round(proba[0] * 100, 2)}%)" if proba is not None else ""
            result = f"❌ Loan Rejected"

        print("\n--- PREDICTION RESULT ---")
        print(result)

        return render_template('index.html', prediction=result, form_data=form)

    except Exception as e:
        print("⚠️ ERROR OCCURRED:", e)
        return render_template(
            'index.html',
            prediction=f"⚠️ Error: {e}",
            form_data=request.form.to_dict()
        )


# ================================
#  RUN FLASK
# ================================
if __name__ == "__main__":
    app.run(debug=True)
