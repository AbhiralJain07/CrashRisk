# CrashRisk — Predictive Aviation Safety Intelligence

A full-stack ML web application that classifies flight scenarios into four risk tiers: **Low, Moderate, High, Critical**.

## Tech Stack
- **Backend**: Python (Flask)
- **ML Model**: Gradient Boosting Classifier (scikit-learn)
- **Frontend**: HTML + CSS + Vanilla JS
- **Explainability**: Feature contribution scoring

## Project Structure
```
crashrisk/
├── app.py              # Flask application + API routes
├── train_model.py      # Model training script
├── model.pkl           # Trained model (auto-generated)
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Full portfolio + simulator UI
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This generates `model.pkl` with ~83% accuracy on 10,000 synthetic scenarios.

### 3. Run the app
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000)

## API Endpoints

### `POST /predict`
Send flight parameters, receive risk tier + probabilities.

```json
{
  "aircraft_age": 20,
  "maintenance_days": 180,
  "pilot_hours": 500,
  "pilot_recency": 30,
  "weather_severity": 7,
  "visibility": 2,
  "turbulence": 6,
  "airport_complexity": 8,
  "time_risk": 1.0,
  "season_risk": 1.0,
  "cargo_type": 0
}
```

**Response:**
```json
{
  "tier": 3,
  "tier_info": { "label": "CRITICAL", "color": "#dc2626", ... },
  "probabilities": { "Low": 1.2, "Moderate": 8.4, "High": 24.1, "Critical": 66.3 },
  "top_factors": [
    { "name": "Pilot Hours", "score": 18.4 },
    ...
  ]
}
```

### `GET /api/model-info`
Returns model metadata and feature importances.

## Risk Features (11 total)
| Feature | Weight |
|---------|--------|
| Aircraft Age | 14% |
| Maintenance Days | 16% |
| Pilot Hours | 18% |
| Pilot Recency | 10% |
| Weather Severity | 15% |
| Visibility | 8% |
| Turbulence | 8% |
| Airport Complexity | 6% |
| Time of Day | 3% |
| Season | 2% |
| Cargo Type | variable |

## Model Details
- **Algorithm**: Gradient Boosting Classifier
- **Trees**: 300 estimators, max depth 5
- **Learning Rate**: 0.08
- **Test Accuracy**: ~83%
- **Training Data**: 10,000 synthetic scenarios with cross-domain interaction terms
