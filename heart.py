from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("heart_model.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get form data
        age = float(request.form["age"])
        sex = float(request.form["sex"])
        cp = float(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = float(request.form["fbs"])
        restecg = float(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = float(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        ca = float(request.form["ca"])
        thal = float(request.form["thal"])

        # Predict
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(features)

        # Result
        result = "Heart Disease Detected!" if prediction[0] == 1 else "No Heart Disease Detected."
        return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)