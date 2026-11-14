import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained KNN model
MODEL_PATH = 'knnmodel.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
else:
    model = None

# Professional, modern HTML layout with a clean dark/light slate color scheme
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KNN Model Predictor</title>
    <style>
        :root {
            --bg-color: #f4f6f9;
            --card-bg: #ffffff;
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --text-main: #1f2937;
            --text-muted: #4b5563;
            --border-color: #e5e7eb;
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
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            width: 100%;
            max-width: 480px;
            border: 1px solid var(--border-color);
        }
        h2 {
            margin-top: 0;
            color: var(--text-main);
            font-size: 24px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 8px;
        }
        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 24px;
        }
        .form-group {
            margin-bottom: 18px;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 500;
            font-size: 14px;
            color: var(--text-main);
        }
        input[type="number"] {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 15px;
            box-sizing: border-box;
            transition: border-color 0.2s;
        }
        input[type="number"]:focus {
            outline: none;
            border-color: var(--primary-color);
        }
        button {
            width: 100%;
            background-color: var(--primary-color);
            color: white;
            padding: 12px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 6px;
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
            text-align: center;
            font-weight: 600;
            font-size: 18px;
        }
        .error-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 6px;
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
    <h2>KNN Prediction Dashboard</h2>
    <p class="subtitle">Enter your features below to get a model prediction</p>
    
    {% if error_msg %}
        <div class="error-box">{{ error_msg }}</div>
    {% endif %}

    <form action="/predict" method="POST">
        <!-- Modify these inputs to match your actual dataset features -->
        <div class="form-group">
            <label for="feature1">Feature 1</label>
            <input type="number" step="any" name="feature1" id="feature1" required value="{{ inputs.get('feature1', '') }}">
        </div>
        <div class="form-group">
            <label for="feature2">Feature 2</label>
            <input type="number" step="any" name="feature2" id="feature2" required value="{{ inputs.get('feature2', '') }}">
        </div>
        <div class="form-group">
            <label for="feature3">Feature 3</label>
            <input type="number" step="any" name="feature3" id="feature3" required value="{{ inputs.get('feature3', '') }}">
        </div>
        
        <button type="submit">Predict Result</button>
    </form>

    {% if prediction is not none %}
        <div class="result-box">
            Prediction Outcome: {{ prediction }}
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
        # Extract features from form values
        f1 = float(request.form.get('feature1', 0))
        f2 = float(request.form.get('feature2', 0))
        f3 = float(request.form.get('feature3', 0))
        
        # Combine inputs into the array structure expected by the model
        input_data = np.array([[f1, f2, f3]])
        
        # Run prediction
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
    # Bind to PORT environment variable for compatibility with cloud hosts like Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
