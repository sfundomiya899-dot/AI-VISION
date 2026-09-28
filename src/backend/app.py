# ============================================================
# AIVision - Artificial Intelligence Platform
# ============================================================

from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import os
import io
import json
import random
import math
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# ------------------------------------------------------------
# APP SETUP
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__,
    template_folder=FRONTEND_DIR,
    static_folder=FRONTEND_DIR,
    static_url_path=''
)

app.config['SECRET_KEY'] = 'aivision-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aivision.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app)
db = SQLAlchemy(app)

# ------------------------------------------------------------
# MATHEMATICS (PM-01)
# ------------------------------------------------------------

class AIMath:
    @staticmethod
    def mean(values):
        return sum(values) / len(values) if values else 0

    @staticmethod
    def std_dev(values):
        if len(values) < 2:
            return 0
        mean = AIMath.mean(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x)) if x > -500 else 0

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def softmax(values):
        exp_vals = np.exp(values - np.max(values))
        return (exp_vals / exp_vals.sum()).tolist()

    @staticmethod
    def bayes(prior, likelihood, evidence):
        return (likelihood * prior) / evidence if evidence != 0 else 0

    @staticmethod
    def correlation(x, y):
        if len(x) != len(y) or len(x) < 2:
            return 0
        mx, my = AIMath.mean(x), AIMath.mean(y)
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(len(x))) / (len(x) - 1)
        sx, sy = AIMath.std_dev(x), AIMath.std_dev(y)
        return cov / (sx * sy) if sx != 0 and sy != 0 else 0

# ------------------------------------------------------------
# MACHINE LEARNING (PM-07)
# ------------------------------------------------------------

class MLModels:
    @staticmethod
    def train_classifier(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, accuracy_score(y_test, model.predict(X_test))

    @staticmethod
    def train_regressor(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, mean_squared_error(y_test, model.predict(X_test))

    @staticmethod
    def train_anomaly(X):
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        return model

    @staticmethod
    def train_clusterer(X, n=3):
        model = KMeans(n_clusters=n, random_state=42, n_init=10)
        model.fit(X)
        return model

# ------------------------------------------------------------
# NEURAL NETWORK (PM-08)
# ------------------------------------------------------------

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = 1 / (1 + np.exp(-np.clip(self.z2, -500, 500)))
        return self.a2

    def train(self, X, y, epochs=100, lr=0.01):
        losses = []
        for _ in range(epochs):
            out = self.forward(X)
            err = out - y
            losses.append(float(np.mean(err ** 2)))

            d_out = err * out * (1 - out)
            d_w2 = np.dot(self.a1.T, d_out)
            d_b2 = np.sum(d_out, axis=0, keepdims=True)

            d_h = np.dot(d_out, self.weights2.T)
            d_h[self.z1 <= 0] = 0
            d_w1 = np.dot(X.T, d_h)
            d_b1 = np.sum(d_h, axis=0, keepdims=True)

            self.weights2 -= lr * d_w2
            self.bias2 -= lr * d_b2
            self.weights1 -= lr * d_w1
            self.bias1 -= lr * d_b1
        return losses

    def predict(self, X):
        return self.forward(X).flatten().tolist()

# ------------------------------------------------------------
# DATABASE MODELS
# ------------------------------------------------------------

class AIData(db.Model):
    __tablename__ = 'ai_data'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    feature1 = db.Column(db.Float)
    feature2 = db.Column(db.Float)
    feature3 = db.Column(db.Float)
    target = db.Column(db.Float)
    category = db.Column(db.String(50))
    source = db.Column(db.String(100))

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'feature1': self.feature1,
            'feature2': self.feature2,
            'feature3': self.feature3,
            'target': self.target,
            'category': self.category,
            'source': self.source
        }

# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/ml')
def ml_page():
    return render_template('ml.html')

@app.route('/deep-learning')
def deep_learning():
    return render_template('deep_learning.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

# ------------------------------------------------------------
# API: GENERATE DATA
# ------------------------------------------------------------

@app.route('/api/data/generate', methods=['POST'])
def generate_data():
    try:
        AIData.query.delete()
        db.session.commit()

        categories = ['Type_A', 'Type_B', 'Type_C']
        for _ in range(500):
            base = random.uniform(0, 100)
            db.session.add(AIData(
                timestamp=datetime.utcnow() - timedelta(hours=random.randint(0, 720)),
                feature1=round(base + random.uniform(-10, 10), 2),
                feature2=round(base * 0.8 + random.uniform(-15, 15), 2),
                feature3=round(base * 1.2 + random.uniform(-5, 5), 2),
                target=round(base * 0.5 + random.uniform(-20, 20), 2),
                category=random.choice(categories),
                source='Generated'
            ))
        db.session.commit()
        return jsonify({'success': True, 'message': 'Generated 500 samples'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

# ------------------------------------------------------------
# API: UPLOAD CSV
# ------------------------------------------------------------

@app.route('/api/data/upload/csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files allowed'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        return _load_dataframe(df, 'CSV: ' + filename)
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

# ------------------------------------------------------------
# API: FETCH FROM URL
# ------------------------------------------------------------

@app.route('/api/data/upload/url', methods=['POST'])
def upload_from_url():
    try:
        url = (request.json or {}).get('url', '').strip()
        if not url:
            return jsonify({'success': False, 'error': 'No URL provided'}), 400
        if not url.startswith('http'):
            return jsonify({'success': False, 'error': 'URL must start with http:// or https://'}), 400

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        return _load_dataframe(df, 'URL')
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

# ------------------------------------------------------------
# HELPER: LOAD DATAFRAME
# ------------------------------------------------------------

def _load_dataframe(df, source):
    AIData.query.delete()
    db.session.commit()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        return jsonify({'success': False, 'error': 'Need at least 3 numeric columns'}), 400

    f1, f2, f3 = numeric_cols[0], numeric_cols[1], numeric_cols[2]
    target = numeric_cols[3] if len(numeric_cols) > 3 else numeric_cols[0]

    category_col = None
    for col in df.columns:
        if col.lower() in ['category', 'class', 'label', 'type', 'species']:
            category_col = col
            break

    cats = ['Type_A', 'Type_B', 'Type_C']
    count = 0

    for _, row in df.head(1000).iterrows():
        try:
            cat = str(row[category_col]) if category_col and pd.notna(row[category_col]) else random.choice(cats)
            db.session.add(AIData(
                timestamp=datetime.utcnow(),
                feature1=float(row[f1]) if pd.notna(row[f1]) else 0,
                feature2=float(row[f2]) if pd.notna(row[f2]) else 0,
                feature3=float(row[f3]) if pd.notna(row[f3]) else 0,
                target=float(row[target]) if pd.notna(row[target]) else 0,
                category=cat,
                source=source
            ))
            count += 1
        except Exception:
            continue

    db.session.commit()
    return jsonify({'success': True, 'message': f'Loaded {count} rows', 'total_rows': count})

# ------------------------------------------------------------
# API: SAMPLE DATASETS
# ------------------------------------------------------------

@app.route('/api/data/samples', methods=['GET'])
def get_samples():
    return jsonify({
        'success': True,
        'datasets': [
            {'name': 'Iris', 'description': 'Flower measurements', 'url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv', 'rows': 150, 'features': 4},
            {'name': 'Tips', 'description': 'Restaurant tips', 'url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv', 'rows': 244, 'features': 7},
            {'name': 'Titanic', 'description': 'Titanic passengers', 'url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv', 'rows': 891, 'features': 15},
            {'name': 'Penguins', 'description': 'Palmer penguins', 'url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv', 'rows': 344, 'features': 7}
        ]
    })

# ------------------------------------------------------------
# API: STATISTICS
# ------------------------------------------------------------

@app.route('/api/math/statistics', methods=['GET'])
def get_statistics():
    data = AIData.query.limit(200).all()
    if not data:
        return jsonify({'success': False, 'error': 'No data available'}), 404

    f1 = [d.feature1 for d in data]
    f2 = [d.feature2 for d in data]
    f3 = [d.feature3 for d in data]
    tg = [d.target for d in data]

    return jsonify({
        'success': True,
        'statistics': {
            'feature1': {'mean': AIMath.mean(f1), 'std_dev': AIMath.std_dev(f1), 'min': min(f1), 'max': max(f1)},
            'feature2': {'mean': AIMath.mean(f2), 'std_dev': AIMath.std_dev(f2), 'min': min(f2), 'max': max(f2)},
            'feature3': {'mean': AIMath.mean(f3), 'std_dev': AIMath.std_dev(f3), 'min': min(f3), 'max': max(f3)},
            'target': {'mean': AIMath.mean(tg), 'std_dev': AIMath.std_dev(tg)},
            'correlations': {
                'f1_f2': AIMath.correlation(f1, f2),
                'f1_f3': AIMath.correlation(f1, f3),
                'f1_target': AIMath.correlation(f1, tg)
            },
            'activation_functions': {
                'sigmoid_demo': AIMath.sigmoid(0.5),
                'relu_demo': AIMath.relu(-1.5),
                'softmax_demo': AIMath.softmax([1.0, 2.0, 3.0]),
                'bayes_demo': AIMath.bayes(0.3, 0.8, 0.5)
            }
        }
    })

# ------------------------------------------------------------
# API: ML MODELS
# ------------------------------------------------------------

@app.route('/api/ml/classify', methods=['POST'])
def ml_classify():
    data = AIData.query.limit(500).all()
    if len(data) < 50:
        return jsonify({'success': False, 'error': 'Need at least 50 samples'}), 400

    X = np.array([[d.feature1, d.feature2, d.feature3] for d in data])
    cats = list(set(d.category for d in data))
    cmap = {c: i for i, c in enumerate(cats)}
    y = np.array([cmap[d.category] for d in data])

    _, acc = MLModels.train_classifier(X, y)
    return jsonify({
        'success': True,
        'model': 'RandomForestClassifier',
        'accuracy': round(acc * 100, 2),
        'categories': cats,
        'samples': len(X)
    })

@app.route('/api/ml/regression', methods=['POST'])
def ml_regression():
    data = AIData.query.limit(500).all()
    if len(data) < 50:
        return jsonify({'success': False, 'error': 'Need at least 50 samples'}), 400

    X = np.array([[d.feature1, d.feature2, d.feature3] for d in data])
    y = np.array([d.target for d in data])

    _, mse = MLModels.train_regressor(X, y)
    return jsonify({
        'success': True,
        'model': 'RandomForestRegressor',
        'mse': round(mse, 2),
        'rmse': round(math.sqrt(mse), 2),
        'samples': len(X)
    })

@app.route('/api/ml/anomaly', methods=['POST'])
def ml_anomaly():
    data = AIData.query.limit(500).all()
    if len(data) < 50:
        return jsonify({'success': False, 'error': 'Need at least 50 samples'}), 400

    X = np.array([[d.feature1, d.feature2, d.feature3] for d in data])
    model = MLModels.train_anomaly(X)
    preds = model.predict(X)
    anomalies = int(np.sum(preds == -1))

    return jsonify({
        'success': True,
        'model': 'IsolationForest',
        'anomalies': anomalies,
        'total': len(X),
        'rate': round(anomalies / len(X) * 100, 2)
    })

@app.route('/api/ml/clustering', methods=['POST'])
def ml_clustering():
    data = AIData.query.limit(500).all()
    if len(data) < 50:
        return jsonify({'success': False, 'error': 'Need at least 50 samples'}), 400

    X = np.array([[d.feature1, d.feature2, d.feature3] for d in data])
    model = MLModels.train_clusterer(X, 3)

    counts = {}
    for label in model.labels_:
        counts[str(label)] = counts.get(str(label), 0) + 1

    return jsonify({
        'success': True,
        'model': 'KMeans',
        'clusters': 3,
        'counts': counts
    })

# ------------------------------------------------------------
# API: DEEP LEARNING
# ------------------------------------------------------------

@app.route('/api/deep-learning/train', methods=['POST'])
def deep_train():
    data = AIData.query.limit(500).all()
    if len(data) < 50:
        return jsonify({'success': False, 'error': 'Need at least 50 samples'}), 400

    X = np.array([[d.feature1, d.feature2, d.feature3] for d in data])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    targets = [d.target for d in data]
    threshold = np.mean(targets)
    y = np.array([[1.0] if t > threshold else [0.0] for t in targets])

    nn = NeuralNetwork(3, 8, 1)
    losses = nn.train(X_scaled, y, epochs=100, lr=0.1)

    preds = nn.predict(X_scaled)
    pred_cls = [1 if p > 0.5 else 0 for p in preds]
    actual = [int(y[i][0]) for i in range(len(y))]
    acc = sum(1 for i in range(len(actual)) if pred_cls[i] == actual[i]) / len(actual)

    return jsonify({
        'success': True,
        'model': 'NeuralNetwork(3-8-1)',
        'epochs': 100,
        'accuracy': round(acc * 100, 2),
        'initial_loss': round(losses[0], 4),
        'final_loss': round(losses[-1], 4),
        'loss_history': [round(l, 4) for l in losses[::10]]
    })

# ------------------------------------------------------------
# API: SUMMARY
# ------------------------------------------------------------

@app.route('/api/analytics/summary', methods=['GET'])
def summary():
    data = AIData.query.limit(500).all()
    if not data:
        return jsonify({'success': False, 'error': 'No data'}), 404

    return jsonify({
        'success': True,
        'total_samples': len(data),
        'categories': list(set(d.category for d in data))
    })

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

with app.app_context():
    db.create_all()
    print("Database created successfully")

if __name__ == '__main__':
    print("\nAIVision running at http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)