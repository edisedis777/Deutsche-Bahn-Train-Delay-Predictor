import sqlite3
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, utcnow

DB_PATH = os.path.join(os.path.dirname(__file__), 'monitor_logs.db')
FEATURES = [
    "weekday", "hour",
    "station_id_BER", "station_id_MUC", "station_id_FRA",
    "train_type_ICE", "train_type_IC", "train_type_RE"
]

def init_monitor_db():
    """Create the SQLite DB if it doesn't exist"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS request_logs (
            timestamp TEXT,
            input_json TEXT,
            prediction REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_request(input_json, prediction):
    """Insert a new request record"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = utcnow().isoformat()
    c.execute('''
        INSERT INTO request_logs (timestamp, input_json, prediction)
        VALUES (?, ?, ?)
    ''', (timestamp, json.dumps(input_json), prediction))
    conn.commit()
    conn.close()

def detect_drift(threshold=0.2):
    """Detect simple input distribution drift using rolling means"""
    conn = sqlite3.connect(DB_PATH)
    df = None
    try:
        df = pd.read_sql_query("SELECT input_json FROM request_logs", conn)
    except sqlite3.Error:
        return
    conn.close()

    if df is None or len(df) < 10:
        return

    X = []
    feature_set = set()
    for raw in df['input_json']:
        try:
            parsed = json.loads(raw)
            feature_set.update(parsed.keys())
            x_vec = list(parsed.values())
            X.append(x_vec)
        except sqlite3.Error:
            continue

    if not X:
        return

    X = np.array(X)
    if X.shape[0] < 10:
        return

    recent = X[-10:]
    historical = X[:-10]

    drift_scores = np.abs(recent.mean(axis=0) - historical.mean(axis=0))

    drift_flags = drift_scores > threshold
    if any(drift_flags):
        print(f"⚠️ Drift detected in {sum(drift_flags)} feature(s):")
        feature_list = list(feature_set)
        for i, flag in enumerate(drift_flags):
            if flag:
                print(f" - {feature_list[i]}")
