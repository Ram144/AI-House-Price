import sys
sys.path.append("libs")  # keep if using portable libs

import os
import csv
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Config ---
DATA_FILE = 'data.csv'
HISTORY_FILE = 'predictions_history.csv'
MODEL_FILE = 'best_model.joblib'
PLOT_FILE = os.path.join('static', 'prediction_plot.png')

# --- Utility functions ---
def ensure_history_file():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp','size_m2','rooms','bathrooms','age','garage','balcony','predicted_k','model','confidence'])

def append_history(entry: dict):
    ensure_history_file()
    with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            entry.get('timestamp'), entry.get('size_m2'), entry.get('rooms'), entry.get('bathrooms'),
            entry.get('age'), entry.get('garage'), entry.get('balcony'), entry.get('predicted_k'),
            entry.get('model'), entry.get('confidence')
        ])

# --- Load dataset ---
print('Loading dataset...')
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found!")

data = pd.read_csv(DATA_FILE)

expected_cols = ['Size_m2','Rooms','Bathrooms','Age','Garage','Balcony','predicted_k']
for c in expected_cols:
    if c not in data.columns:
        raise ValueError(f"Missing column: {c}")

feature_cols = ['Size_m2','Rooms','Bathrooms','Age','Garage','Balcony']
X = data[feature_cols]
y = data['predicted_k']

# --- Train multiple models ---
print('Training models...')
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
}

cv_scores = {}
for name, m in models.items():
    try:
        scores = cross_val_score(m, X, y, cv=5, scoring='r2')
        cv_scores[name] = np.mean(scores)
        print(f"{name} CV R2 mean = {cv_scores[name]:.4f}")
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

best_name = max(cv_scores, key=cv_scores.get)
best_model = models[best_name]
best_model.fit(X, y)
joblib.dump({'model': best_model, 'name': best_name}, MODEL_FILE)
print(f"Best model: {best_name}, saved as {MODEL_FILE}")

# --- Confidence calculation ---
def compute_confidence(model, x_array):
    try:
        if hasattr(model, 'estimators_'):
            preds = np.array([est.predict(x_array) for est in model.estimators_])
            std = preds.std(axis=0)[0]
            mean_pred = np.mean(preds)
            conf = max(0.0, 100.0 - (std / (abs(mean_pred) + 1e-6)) * 100.0)
            return round(conf, 2)
        else:
            residuals = (y - model.predict(X)).std()
            pred = model.predict(x_array)[0]
            conf = max(0.0, 100.0 - (residuals / (abs(pred) + 1e-6)) * 100.0)
            return round(conf, 2)
    except Exception as e:
        print('Confidence calc error:', e)
        return 50.0

# --- Smart textual summary ---
def smart_summary(size, rooms, bathrooms, age, garage, balcony, predicted_k):
    parts = []
    parts.append(f"The model predicts ~{predicted_k:.1f}k$ for a {size} m² house with {rooms} rooms.")
    if age <= 5:
        parts.append("The house is relatively new which increases its value.")
    elif age >= 30:
        parts.append("Older house; renovation may reduce value.")
    if bathrooms >= 2:
        parts.append("Multiple bathrooms positively influence price.")
    if garage:
        parts.append("Having a garage adds value.")
    if balcony:
        parts.append("A balcony slightly increases price.")
    return ' '.join(parts)

# --- Plot ---
def generate_plot(add_point=None):
    plt.figure(figsize=(8,5))
    plt.scatter(X['Size_m2'], y, color='blue', label='Real Prices')
    idx = np.argsort(X['Size_m2'])
    plt.plot(X['Size_m2'].iloc[idx], best_model.predict(X.iloc[idx]), color='red', label='Model Prediction')
    if add_point:
        plt.scatter([add_point['size']], [add_point['pred']], color='green', s=100, label='Your Prediction')
    plt.xlabel('Size (m²)')
    plt.ylabel('Price (k$)')
    plt.title('House Price Predictions')
    plt.legend()
    plt.tight_layout()
    os.makedirs('static', exist_ok=True)
    plt.savefig(PLOT_FILE)
    plt.close()

generate_plot()

# --- Flask App ---
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    price = None
    summary = None
    confidence = None
    if request.method == 'POST':
        try:
            size = float(request.form.get('size', 100))
            rooms = int(request.form.get('rooms', 3))
            bathrooms = int(request.form.get('bathrooms', 1))
            age = int(request.form.get('age', 10))
            garage = 1 if request.form.get('garage') == 'on' else 0
            balcony = 1 if request.form.get('balcony') == 'on' else 0
            X_new = pd.DataFrame([[size, rooms, bathrooms, age, garage, balcony]], columns=feature_cols)
            pred = best_model.predict(X_new)[0]
            confidence = compute_confidence(best_model, X_new.values)
            summary = smart_summary(size, rooms, bathrooms, age, garage, balcony, pred)
            append_history({
                'timestamp': datetime.utcnow().isoformat(),
                'size_m2': size, 'rooms': rooms, 'bathrooms': bathrooms,
                'age': age, 'garage': garage, 'balcony': balcony,
                'predicted_k': round(pred,3), 'model': best_name, 'confidence': confidence
            })
            generate_plot(add_point={'size': size, 'pred': pred})
            price = round(pred,3)
        except Exception as e:
            print('Predict error:', e)
            price = 'Error'
    # History last 10
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
            history = rows[-10:][::-1]
    return render_template('index.html', price=price, summary=summary, confidence=confidence, history=history)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(force=True)
    try:
        size = float(payload.get('size',100))
        rooms = int(payload.get('rooms',3))
        bathrooms = int(payload.get('bathrooms',1))
        age = int(payload.get('age',10))
        garage = 1 if payload.get('garage',False) else 0
        balcony = 1 if payload.get('balcony',False) else 0
        X_new = pd.DataFrame([[size, rooms, bathrooms, age, garage, balcony]], columns=feature_cols)
        pred = best_model.predict(X_new)[0]
        conf = compute_confidence(best_model, X_new.values)
        summary = smart_summary(size, rooms, bathrooms, age, garage, balcony, pred)
        append_history({
            'timestamp': datetime.utcnow().isoformat(),
            'size_m2': size, 'rooms': rooms, 'bathrooms': bathrooms,
            'age': age, 'garage': garage, 'balcony': balcony,
            'predicted_k': round(pred,3), 'model': best_name, 'confidence': conf
        })
        generate_plot(add_point={'size': size, 'pred': pred})
        return jsonify({'prediction_k': round(pred,3), 'confidence': conf, 'model': best_name, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    ensure_history_file()
    app.run(host='0.0.0.0', port=5000, debug=True)

