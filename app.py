from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    payload = pickle.load(f)

model = payload['model']
FEATURES = payload['features']
MODEL_ACCURACY = payload['accuracy']
FEATURE_IMPORTANCES = payload['feature_importances']

RISK_TIERS = {
    0: {'label': 'LOW',      'color': '#22c55e', 'bg': '#052e16', 'emoji': '✅', 'desc': 'Flight conditions are within normal safety parameters. Standard protocols apply.'},
    1: {'label': 'MODERATE', 'color': '#f59e0b', 'bg': '#451a03', 'emoji': '⚠️', 'desc': 'Elevated risk factors detected. Enhanced monitoring and crew briefing recommended.'},
    2: {'label': 'HIGH',     'color': '#f97316', 'bg': '#431407', 'emoji': '🔴', 'desc': 'Multiple risk factors converging. Operational review and possible delay advised.'},
    3: {'label': 'CRITICAL', 'color': '#dc2626', 'bg': '#450a0a', 'emoji': '🚨', 'desc': 'Critical risk level. Immediate safety review required before any flight operations.'},
}

@app.route('/')
def index():
    return render_template('index.html',
                           model_accuracy=round(MODEL_ACCURACY * 100, 2),
                           feature_importances=json.dumps(FEATURE_IMPORTANCES))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            float(data['aircraft_age']),
            float(data['maintenance_days']),
            float(data['pilot_hours']),
            float(data['pilot_recency']),
            float(data['weather_severity']),
            float(data['visibility']),
            float(data['turbulence']),
            float(data['airport_complexity']),
            float(data['time_risk']),
            float(data['season_risk']),
            float(data['cargo_type']),
        ]
        import pandas as pd; X = pd.DataFrame([features], columns=FEATURES)
        tier = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()

        # Compute SHAP-like factor contributions (use feature importances × normalized input)
        fi = model.feature_importances_
        norms = [
            features[0]/45, features[1]/365, 1-(features[2]/25000)**0.4,
            features[3]/120, features[4]/10, 1-(features[5]/15),
            features[6]/10, features[7]/10, features[8], features[9], features[10]/2
        ]
        contributions = {FEATURES[i]: round(float(fi[i] * norms[i]), 4) for i in range(len(FEATURES))}
        top_factors = sorted(contributions.items(), key=lambda x: -x[1])[:5]

        return jsonify({
            'tier': tier,
            'tier_info': RISK_TIERS[tier],
            'probabilities': {
                'Low': round(proba[0]*100, 1),
                'Moderate': round(proba[1]*100, 1),
                'High': round(proba[2]*100, 1),
                'Critical': round(proba[3]*100, 1),
            },
            'top_factors': [{'name': k.replace('_', ' ').title(), 'score': round(v*100, 1)} for k, v in top_factors]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model-info')
def model_info():
    return jsonify({
        'accuracy': round(MODEL_ACCURACY * 100, 2),
        'algorithm': 'Gradient Boosting Classifier',
        'training_samples': 10000,
        'features': len(FEATURES),
        'risk_tiers': 4,
        'feature_importances': {k: round(v*100, 2) for k, v in FEATURE_IMPORTANCES.items()}
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
