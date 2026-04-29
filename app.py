from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/diabetes_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [float(request.form[key]) for key in request.form]
            final_features = np.array([features])
            prediction = model.predict(final_features)
            result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        except ValueError:
            result = "Invalid input. Please enter all numbers."
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
