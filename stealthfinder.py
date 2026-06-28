import pandas as pd
import numpy as np


data = pd.read_csv("Network_traffic.csv")
data.columns = data.columns.str.strip().str.lower()


data['time_dt'] = pd.to_datetime(data['time'], errors='coerce')
data['time_sec'] = (data['time_dt'] - data['time_dt'].min()).dt.total_seconds().fillna(0)
data['length'] = pd.to_numeric(data['length'], errors='coerce').fillna(0)
data['protocol_encoded'] = data['protocol'].astype(str).astype('category').cat.codes

data['is_anomaly'] = data['info'].astype(str).str.lower().str.contains('request|ping|nmap|scan|attack', regex=True).astype(float)

X_raw = data[['time_sec', 'length', 'protocol_encoded']].to_numpy().astype(float)
X = (X_raw - np.mean(X_raw, axis=0)) / (np.std(X_raw, axis=0) + 1e-8)


W1 = np.random.randn(3, 8) * 0.1
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 1) * 0.1
b2 = np.zeros((1, 1))

Z1 = np.dot(X, W1) + b1
A1 = np.maximum(0, Z1)
Z2 = np.dot(A1, W2) + b2
predictions = 1 / (1 + np.exp(-np.clip(Z2, -500, 500)))


data['ai_predicted'] = (predictions > 0.5).astype(float)

stealth_packets = data[(data['is_anomaly'] == 0) & (data['ai_predicted'] == 1)]

print(f"=== DETECTED STEALTH PACKETS: {len(stealth_packets)} ===")
if len(stealth_packets) > 0:
    print(stealth_packets[['time', 'source', 'destination', 'protocol', 'length', 'info']].head(10))
else:
    print("No stealth packets detected")