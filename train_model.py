import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

np.random.seed(42)
N = 10000

def generate_dataset(n):
    aircraft_age        = np.random.exponential(scale=12, size=n).clip(0, 45)
    maintenance_days    = np.random.exponential(scale=60, size=n).clip(0, 365)
    pilot_hours         = np.random.lognormal(mean=8.2, sigma=0.9, size=n).clip(200, 25000)
    pilot_recency       = np.random.exponential(scale=15, size=n).clip(0, 120)
    weather_severity    = np.random.beta(a=1.5, b=3, size=n) * 10
    visibility          = (15 - weather_severity * 1.2 + np.random.normal(0, 1, n)).clip(0.1, 15)
    turbulence          = (weather_severity * 0.7 + np.random.beta(1, 4, n) * 4).clip(0, 10)
    airport_complexity  = np.random.beta(2, 2, n) * 10
    hour                = np.random.randint(0, 24, n)
    time_risk           = np.where((hour >= 22) | (hour <= 5), 1.0,
                          np.where((hour >= 6) & (hour <= 9), 0.4, 0.2))
    season              = np.random.randint(0, 4, n)
    season_risk         = np.where(season == 3, 1.0, np.where(season == 1, 0.6, 0.3))
    cargo_type          = np.random.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1])

    age_norm   = aircraft_age / 45
    maint_norm = maintenance_days / 365
    pilot_norm = 1 - (pilot_hours / 25000) ** 0.4
    rec_norm   = pilot_recency / 120
    weath_norm = weather_severity / 10
    vis_norm   = 1 - (visibility / 15)
    turb_norm  = turbulence / 10
    air_norm   = airport_complexity / 10

    raw = (age_norm*0.14 + maint_norm*0.16 + pilot_norm*0.18 + rec_norm*0.10 +
           weath_norm*0.15 + vis_norm*0.08 + turb_norm*0.08 + air_norm*0.06 +
           time_risk*0.03 + season_risk*0.02)
    raw += age_norm*weath_norm*0.08 + pilot_norm*weath_norm*0.06
    raw += maint_norm*age_norm*0.05 + time_risk*weath_norm*0.04
    raw += np.random.normal(0, 0.04, n)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

    risk_tier = np.where(raw < 0.38, 0,
                np.where(raw < 0.65, 1,
                np.where(raw < 0.85, 2, 3)))

    return pd.DataFrame({
        'aircraft_age': aircraft_age, 'maintenance_days': maintenance_days,
        'pilot_hours': pilot_hours, 'pilot_recency': pilot_recency,
        'weather_severity': weather_severity, 'visibility': visibility,
        'turbulence': turbulence, 'airport_complexity': airport_complexity,
        'time_risk': time_risk, 'season_risk': season_risk,
        'cargo_type': cargo_type, 'risk_tier': risk_tier
    })

print("Generating dataset...")
df = generate_dataset(N)
print(f"Distribution: { {i: int((df.risk_tier==i).sum()) for i in range(4)} }")

FEATURES = ['aircraft_age','maintenance_days','pilot_hours','pilot_recency',
            'weather_severity','visibility','turbulence','airport_complexity',
            'time_risk','season_risk','cargo_type']

X, y = df[FEATURES], df['risk_tier']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training model...")
model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.08, max_depth=5,
                                    min_samples_split=20, subsample=0.85, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, model.predict(X_test), target_names=['Low','Moderate','High','Critical']))

importances = dict(zip(FEATURES, model.feature_importances_))
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': FEATURES, 'accuracy': acc, 'feature_importances': importances}, f)
print("Saved model.pkl")
