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

# Premium aesthetic matched precisely to the provided reference UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Explore Health Insurance - KNN</title>
    <style>
        body {
            margin: 0;
            padding: 40px 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #f3a39e 0%, #da6d81 50%, #688fa7 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            box-sizing: border-box;
        }
        .wrapper {
            position: relative;
            width: 100%;
            max-width: 600px;
        }
        .badge {
            position: absolute;
            top: -14px;
            left: 20px;
            background-color: #008753;
            color: #ffffff;
            font-size: 13px;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 6px;
            z-index: 10;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .badge::after {
            content: '';
            position: absolute;
            bottom: -5px;
            left: 10px;
            border-width: 5px 5px 0 0;
            border-style: solid;
            border-color: #00643d transparent transparent transparent;
        }
        .container {
            background: rgba(43, 41, 46, 0.75);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 40px 32px 32px 32px;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.25);
            color: #ffffff;
            box-sizing: border-box;
        }
        h2 {
            margin: 12px 0 6px 0;
            font-size: 26px;
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        p.subtitle {
            color: #d1d1d6;
            font-size: 15px;
            margin: 0 0 28px 0;
            opacity: 0.9;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        @media (max-width: 550px) {
            .grid { grid-template-columns: 1fr; }
            .full-width { grid-column: span 1 !important; }
        }
        .form-group {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 14px;
            padding: 10px 16px;
            box-sizing: border-box;
            transition: border-color 0.2s, background-color 0.2s;
        }
        .form-group:focus-within {
            border-color: rgba(255, 255, 255, 0.4);
            background: rgba(255, 255, 255, 0.1);
        }
        .full-width {
            grid-column: span 2;
        }
        label {
            display: block;
            font-size: 13px;
            color: #ffffff;
            font-weight: 600;
            margin-bottom: 2px;
        }
        input[type="number"] {
            width: 100%;
            background: transparent;
            border: none;
            outline: none;
            color: #ffffff;
            font-size: 15px;
            padding: 4px 0;
            box-sizing: border-box;
        }
        input[type="number"]::placeholder {
            color: #a1a1aa;
            opacity: 0.7;
        }
        .checkbox-container {
            display: flex;
            align-items: center;
            margin-top: 20px;
            font-size: 13px;
            color: #e4e4e7;
            cursor: pointer;
        }
        .checkbox-container input {
            margin-right: 10px;
            accent-color: #ffffff;
            width: 16px;
            height: 16px;
        }
        button {
            width: 100%;
            background-color: #c2c3c9;
            color: #1c1917;
            padding: 15px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 24px;
        }
        button:hover {
            background-color: #ffffff;
        }
        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 12px;
            background-color: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.3);
            color: #4ade80;
            text-align: center;
            font-weight: 600;
            font-size: 17px;
        }
        .error-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 12px;
            background-color: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #f87171;
            text-align: center;
            font-weight: 600;
        }
    </style>
</head>
<body>

<div class="wrapper">
    <div class="badge">Buy Now with 0% GST</div>
    
    <div class="container">
        <h2>Explore Health Insurance</h2>
        <p class="subtitle">Insure your Loved Ones in less than 5 mins!</p>
        
        {% if error_msg %}
            <div class="error-box">{{ error_msg }}</div>
        {% endif %}

        <form action="/predict" method="POST">
            <div class="grid">
                <div class="form-group">
                    <label for="Age">Age</label>
                    <input type="number" name="Age" id="Age" placeholder="Enter Age value" required value="{{ inputs.get('Age', '') }}">
                </div>
                
                <div class="form-group">
                    <label for="Sex">Sex</label>
                    <input type="number" name="Sex" id="Sex" placeholder="Encoded numeric" required value="{{ inputs.get('Sex', '') }}">
                </div>

                <div class="form-group">
                    <label for="ChestPainType">Chest Pain Type</label>
                    <input type="number" name="ChestPainType" id="ChestPainType" placeholder="Encoded status value" required value="{{ inputs.get('ChestPainType', '') }}">
                </div>

                <div class="form-group">
                    <label for="RestingBP">Resting Blood Pressure</label>
                    <input type="number" name="RestingBP" id="RestingBP" placeholder="mm Hg" required value="{{ inputs.get('RestingBP', '') }}">
                </div>

                <div class="form-group">
                    <label for="Cholesterol">Cholesterol</label>
                    <input type="number" name="Cholesterol" id="Cholesterol" placeholder="mm/dl" required value="{{ inputs.get('Cholesterol', '') }}">
                </div>

                <div class="form-group">
                    <label for="FastingBS">Fasting Blood Sugar</label>
                    <input type="number" name="FastingBS" id="FastingBS" placeholder="Encoded check value" required value="{{ inputs.get('FastingBS', '') }}">
                </div>

                <div class="form-group">
                    <label for="RestingECG">Resting ECG Results</label>
                    <input type="number" name="RestingECG" id="RestingECG" placeholder="Encoded numeric" required value="{{ inputs.get('RestingECG', '') }}">
                </div>

                <div class="form-group">
                    <label for="MaxHR">Max Heart Rate</label>
                    <input type="number" name="MaxHR" id="MaxHR" placeholder="Max rate achieved" required value="{{ inputs.get('MaxHR', '') }}">
                </div>

                <div class="form-group">
                    <label for="ExerciseAngina">Exercise Induced Angina</label>
                    <input type="number" name="ExerciseAngina" id="ExerciseAngina" placeholder="Encoded response" required value="{{ inputs.get('ExerciseAngina', '') }}">
                </div>

                <div class="form-group">
                    <label for="Oldpeak">Oldpeak</label>
                    <input type="number" step="any" name="Oldpeak" id="Oldpeak" placeholder="ST depression value" required value="{{ inputs.get('Oldpeak', '') }}">
                </div>

                <div class="form-group full-width">
                    <label for="ST_Slope">ST Slope</label>
                    <input type="number" name="ST_Slope" id="ST_Slope" placeholder="Encoded metric slope" required value="{{ inputs.get('ST_Slope', '') }}">
                </div>
            </div>
            
            <label class="checkbox-container">
                <input type="checkbox" required checked>
                By proceeding, you agree to the Terms & Conditions
            </label>
            
            <button type="submit">Get Quote</button>
        </form>

        {% if prediction is not none %}
            <div class="result-box">
                Diagnostic Result Class Output: {{ prediction }}
            </div>
        {% endif %}
    </div>
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
