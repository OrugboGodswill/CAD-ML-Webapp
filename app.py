from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import joblib
import os

app = Flask(__name__, template_folder='templates', static_folder='static')


# Load the pipeline (model + scaler together)
try:
    model_path = os.path.join(os.path.dirname(__file__), "cad_pipeline.pkl")
    model = joblib.load(open(model_path, "rb"))
    print("Pipeline loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test_form():
    return render_template("test.html")

@app.route('/about')
def about_page():
    return render_template("about.html")

@app.route('/Doc')
def Doc_page():
    return render_template("doc.html")

@app.route('/statchart')
def statchart():
    return render_template("statchart.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input
        sex = int(request.form['sex'])
        systolic_bp = float(request.form['systolic_bp'])
        tobacco_per_yer = float(request.form['tobacco_per_yer'])
        chol = float(request.form['chol'])
        adiposity = float(request.form['adiposity'])
        famhist = int(request.form['famhist'])   # 0 or 1
        stress_prone = int(request.form['stress_prone'])
        Bmi = float(request.form['Bmi'])
        alco_per_year = float(request.form['alco_per_year'])
        age = float(request.form['age'])

        # Prepare features
        features = np.array([[sex, systolic_bp, tobacco_per_yer, chol, adiposity,
                              famhist, stress_prone, Bmi, alco_per_year, age]])

        # Predict
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1] * 100  # probability of CAD

        if prediction == 1:
            result = f"⚠️ High Risk of CAD. Probability: {prob:.2f}%"
        else:
            result = f"✅ Low Risk of CAD. Probability: {prob:.2f}%"

        return render_template("test.html", prediction=result)

    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)