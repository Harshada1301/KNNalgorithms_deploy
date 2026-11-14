import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained KNN model
MODEL_PATH = 'knnmodel.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
else:
    model = None

# Custom dashboard tailored for the 11 heart disease dataset features
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Predictor (KNN)</title>
    <style>
        :root {
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --text-main: #0f172a;
            --text-muted: #475569;
            --border-color: #cbd5e1;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            box-sizing: border-box;
        }
        .container {
            background: var(--card-bg);
            padding: 32px;
            border-radius: 16px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 650px;
            border: 1px solid #e2e8f0;
        }
        h2 {
            margin-top: 0;
            color: var(--text-main);
            font-size: 26px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 8px;
        }
        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 32px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        @media (max-width: 500px) {
            .grid { grid-template-columns: 1fr; }
        }
        .form-group {
            margin-bottom: 12px;
        }
        .full-width {
            grid-column: span 2;
        }
        @media (max-width: 500px) {
            .full-width { grid-column: span 1; }
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 500;
            font-size: 13px;
            color: var(--text-muted);
        }
        input[type="number"], select {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 14px;
            box-sizing: border-box;
            background-color: #fff;
            transition: border-color 0.2s;
        }
        input[type="number"]:focus, select:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        button {
            width: 100%;
            background-color: var(--primary-color);
            color: white;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 20px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-box {
            margin-top: 28px;
            padding: 16px;
            border-radius: 8px;
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
            text-align: center;
            font-weight: 600;
            font-size: 18px;
        }
        .error-box {
            margin-top: 28px;
            padding: 16px;
            border-radius: 8px;
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            text-align: center;
            font-weight: 600;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Heart Disease Predictor</h2>
    <p class="subtitle">Provide patient clinical metrics to calculate KNN classification outcome.</p>
    
    {% if error_msg %}
        <div class="error-box">{{ error_msg }}</div>
    {% endif %}

    <form action="/predict" method="POST">
        <div class="grid">
            <div class="form-group">
                <label for="Age">Age</label>
                <input type="number" name="Age" id="Age" required value="{{ inputs.get('Age', '') }}">
            </div>
            
            <div class="form-group">
                <label for="Sex">Sex (Encoded numeric)</label>
                <input type="number" name="Sex" id="Sex" required value="{{ inputs.get('Sex', '') }}">
            </div>

            <div class="form-group">
                <label for="ChestPainType">Chest Pain Type (Encoded)</label>
                <input type="number" name="ChestPainType" id="ChestPainType" required value="{{ inputs.get('ChestPainType', '') }}">
            </div>

            <div class="form-group">
                <label for="RestingBP">Resting Blood Pressure (mm Hg)</label>
                <input type="number" name="RestingBP" id="RestingBP" required value="{{ inputs.get('RestingBP', '') }}">
            </div>

            <div class="form-group">
                <label for="Cholesterol">Serum Cholesterol (mm/dl)</label>
                <input type="number" name="Cholesterol" id="Cholesterol" required value="{{ inputs.get('Cholesterol', '') }}">
            </div>

            <div class="form-group">
                <label for="FastingBS">Fasting Blood Sugar (Encoded)</label>
                <input type="number" name="FastingBS" id="FastingBS" required value="{{ inputs.get('FastingBS', '') }}">
            </div>

            <div class="form-group">
                <label for="RestingECG">Resting ECG Results (Encoded)</label>
                <input type="number" name="RestingECG" id="RestingECG" required value="{{ inputs.get('RestingECG', '') }}">
            </div>

            <div class="form-group">
                <label for="MaxHR">Max Heart Rate Achieved</label>
                <input type="number" name="MaxHR" id="MaxHR" required value="{{ inputs.get('MaxHR', '') }}">
            </div>

            <div class="form-group">
                <label for="ExerciseAngina">Exercise Induced Angina (Encoded)</label>
                <input type="number" name="ExerciseAngina" id="ExerciseAngina" required value="{{ inputs.get('ExerciseAngina', '') }}">
            </div>

            <div class="form-group">
                <label for="Oldpeak">Oldpeak (ST Depression value)</label>
                <input type="number" step="any" name="Oldpeak" id="Oldpeak" required value="{{ inputs.get('Oldpeak', '') }}">
            </div>

            <div class="form-group full-width">
                <label for="ST_Slope">ST Slope (Encoded)</label>
                <input type="number" name="ST_Slope" id="ST_Slope" required value="{{ inputs.get('ST_Slope', '') }}">
            </div>
        </div>
        
        <button type="submit">Run Diagnostics</button>
    </form>

    {% if prediction is not none %}
        <div class="result-box">
            Model Prediction Class Output: {{ prediction }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error_msg="Warning: 'knnmodel.pkl' not found on server.", inputs={})
    return render_template_string(HTML_TEMPLATE, prediction=None, inputs={})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error_msg="Error: Model file is missing.", inputs={})
    
    try:
        # Extract all 11 features precisely matching the structural requirements
        features = [
            float(request.form.get('Age', 0)),
            float(request.form.get('Sex', 0)),
            float(request.form.get('ChestPainType', 0)),
            float(request.form.get('RestingBP', 0)),
            float(request.form.get('Cholesterol', 0)),
            float(request.form.get('FastingBS', 0)),
            float(request.form.get('RestingECG', 0)),
            float(request.form.get('MaxHR', 0)),
            float(request.form.get('ExerciseAngina', 0)),
            float(request.form.get('Oldpeak', 0)),
            float(request.form.get('ST_Slope', 0))
        ]
        
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        
        return render_template_string(
            HTML_TEMPLATE, 
            prediction=prediction, 
            inputs=request.form
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE, 
            error_msg=f"Prediction failed: {str(e)}", 
            inputs=request.form
        )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
