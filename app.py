import os
import pickle

import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knnmodel.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

# Label encodings (alphabetical, as produced by sklearn LabelEncoder / pandas category codes)
CHOICES = {
    "Sex": [(0, "Female"), (1, "Male")],
    "ChestPainType": [(0, "ASY – Asymptomatic"), (1, "ATA – Atypical Angina"),
                      (2, "NAP – Non-Anginal Pain"), (3, "TA – Typical Angina")],
    "FastingBS": [(0, "≤ 120 mg/dl"), (1, "> 120 mg/dl")],
    "RestingECG": [(0, "LVH – Left Ventricular Hypertrophy"), (1, "Normal"), (2, "ST – ST-T Abnormality")],
    "ExerciseAngina": [(0, "No"), (1, "Yes")],
    "ST_Slope": [(0, "Down"), (1, "Flat"), (2, "Up")],
}

NUMERIC = {
    "Age": dict(label="Age (years)", min=1, max=120, step=1, ph="e.g. 54"),
    "RestingBP": dict(label="Resting BP (mm Hg)", min=0, max=300, step=1, ph="e.g. 130"),
    "Cholesterol": dict(label="Cholesterol (mg/dl)", min=0, max=700, step=1, ph="e.g. 240"),
    "MaxHR": dict(label="Max Heart Rate (bpm)", min=40, max=250, step=1, ph="e.g. 150"),
    "Oldpeak": dict(label="Oldpeak (ST depression)", min=-5, max=10, step=0.1, ph="e.g. 1.2"),
}

LABELS = {
    "Sex": "Sex", "ChestPainType": "Chest Pain Type", "FastingBS": "Fasting Blood Sugar",
    "RestingECG": "Resting ECG", "ExerciseAngina": "Exercise-Induced Angina", "ST_Slope": "ST Slope",
}

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heart Disease Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0b1d2a; --bg2:#10293b; --card:#ffffff; --ink:#12263a; --muted:#64788b;
    --line:#dbe4ec; --teal:#14b8a6; --teal-d:#0d9488; --coral:#ef5b5b; --coral-d:#d63f3f;
    --ok:#16a34a; --ok-bg:#e8f8ee; --bad-bg:#fdeaea;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:Inter,system-ui,sans-serif;color:var(--ink);
    background:radial-gradient(1200px 600px at 10% -10%,#17425a 0%,transparent 60%),
               radial-gradient(900px 500px at 110% 10%,#0e5a5a 0%,transparent 55%),var(--bg);
    min-height:100vh;padding:32px 16px}
  .wrap{max-width:860px;margin:0 auto}
  header{color:#fff;text-align:center;margin-bottom:24px}
  header .pulse{display:inline-flex;align-items:center;justify-content:center;width:56px;height:56px;
    border-radius:50%;background:rgba(239,91,91,.15);font-size:28px;margin-bottom:10px;
    animation:beat 1.6s infinite}
  @keyframes beat{0%,100%{transform:scale(1)}15%{transform:scale(1.12)}30%{transform:scale(1)}45%{transform:scale(1.08)}}
  h1{font-family:'DM Serif Display',serif;font-weight:400;font-size:2.3rem;margin:0 0 6px}
  header p{margin:0;color:#a9c0cf;font-size:.98rem}
  .card{background:var(--card);border-radius:18px;padding:28px;box-shadow:0 20px 50px rgba(0,0,0,.35)}
  .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px 22px}
  @media(max-width:640px){.grid{grid-template-columns:1fr}.card{padding:20px}}
  label{display:block;font-size:.8rem;font-weight:600;color:var(--muted);text-transform:uppercase;
    letter-spacing:.05em;margin-bottom:6px}
  input,select{width:100%;padding:12px 14px;border:1.5px solid var(--line);border-radius:10px;
    font:inherit;color:var(--ink);background:#f8fafc;transition:.15s}
  input:focus,select:focus{outline:none;border-color:var(--teal);background:#fff;box-shadow:0 0 0 4px rgba(20,184,166,.15)}
  button{margin-top:26px;width:100%;padding:15px;border:0;border-radius:12px;font:600 1.05rem Inter,sans-serif;
    color:#fff;background:linear-gradient(135deg,var(--teal),var(--teal-d));cursor:pointer;transition:.15s}
  button:hover{transform:translateY(-1px);box-shadow:0 10px 22px rgba(13,148,136,.35)}
  .result{margin-top:24px;padding:20px 22px;border-radius:14px;display:flex;gap:16px;align-items:center}
  .result.bad{background:var(--bad-bg);border:1.5px solid #f6b9b9}
  .result.ok{background:var(--ok-bg);border:1.5px solid #a8e3bd}
  .result .ico{font-size:2rem}
  .result h2{margin:0 0 4px;font-size:1.2rem}
  .result.bad h2{color:var(--coral-d)} .result.ok h2{color:var(--ok)}
  .result p{margin:0;color:var(--muted);font-size:.92rem}
  .err{margin-top:20px;padding:14px;border-radius:10px;background:var(--bad-bg);color:var(--coral-d)}
  footer{text-align:center;color:#7f98a9;font-size:.8rem;margin-top:18px;line-height:1.5}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="pulse">❤️</div>
    <h1>Heart Disease Predictor</h1>
    <p>Enter patient details to estimate heart disease risk with a K-Nearest Neighbors model.</p>
  </header>

  <div class="card">
    <form method="post" action="/">
      <div class="grid">
        {% for name, c in numeric.items() %}
        <div>
          <label for="{{name}}">{{c.label}}</label>
          <input id="{{name}}" name="{{name}}" type="number" required
                 min="{{c.min}}" max="{{c.max}}" step="{{c.step}}" placeholder="{{c.ph}}"
                 value="{{values.get(name,'')}}">
        </div>
        {% endfor %}
        {% for name, opts in choices.items() %}
        <div>
          <label for="{{name}}">{{labels[name]}}</label>
          <select id="{{name}}" name="{{name}}" required>
            {% for v, text in opts %}
            <option value="{{v}}" {{'selected' if values.get(name)|string == v|string}}>{{text}}</option>
            {% endfor %}
          </select>
        </div>
        {% endfor %}
      </div>
      <button type="submit">Predict</button>
    </form>

    {% if error %}<div class="err">{{error}}</div>{% endif %}

    {% if result is not none %}
    <div class="result {{'bad' if result==1 else 'ok'}}">
      <div class="ico">{{'⚠️' if result==1 else '✅'}}</div>
      <div>
        <h2>{{'Higher risk of heart disease' if result==1 else 'Lower risk of heart disease'}}</h2>
        <p>{% if prob is not none %}Model confidence: {{prob}}% · {% endif %}This is a statistical estimate, not a medical diagnosis.</p>
      </div>
    </div>
    {% endif %}
  </div>
  <footer>For educational use only. Consult a qualified healthcare professional for medical advice.</footer>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result, prob, error, values = None, None, None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            row = {f: float(values[f]) for f in FEATURES}
            X = pd.DataFrame([row], columns=FEATURES)
            result = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            prob = round(float(proba[list(model.classes_).index(result)]) * 100, 1)
        except (KeyError, ValueError):
            error = "Please fill in every field with valid numbers."
    return render_template_string(
        PAGE, numeric=NUMERIC, choices=CHOICES, labels=LABELS,
        values=values, result=result, prob=prob, error=error,
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
