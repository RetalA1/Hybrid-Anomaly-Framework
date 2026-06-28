import pandas as pd
import numpy as np

data = pd.read_csv("Network_traffic.csv")
data.columns = data.columns.str.strip().str.lower()

data['time'] = pd.to_datetime(data['time'], errors='coerce')
data['time'] = (data['time'] - data['time'].min()).dt.total_seconds().fillna(0)

data['length'] = pd.to_numeric(data['length'], errors='coerce').fillna(0)

data['protocol_encoded'] = data['protocol'].astype(str).astype('category').cat.codes

data['is_anomaly'] = data['info'].astype(str).str.lower().str.contains('request|ping|nmap|scan|attack', regex=True).astype(float)

X_raw = data[['time', 'length', 'protocol_encoded']].to_numpy().astype(float)
Y = data['is_anomaly'].to_numpy().reshape(-1, 1).astype(float)

X = (X_raw - np.mean(X_raw, axis=0)) / (np.std(X_raw, axis=0) + 1e-8)

shuffled_indices = np.random.permutation(len(X))
X = X[shuffled_indices]
Y = Y[shuffled_indices]

split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

class AnomalyDetectorNET:
    def __init__(self, input_dim, hidden_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.001
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.001
        self.b2 = np.zeros((1, 1))

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def _sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.Y_hat = self._sigmoid(self.Z2)
        return self.Y_hat

    def compute_loss(self, Y, Y_hat):
        m = Y.shape[0]
        epsilon = 1e-15
        Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
        loss = - (1 / m) * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return loss

    def backward(self, X, Y):
        m = Y.shape[0]
        self.dZ2 = self.Y_hat - Y
        self.dW2 = (1 / m) * np.dot(self.A1.T, self.dZ2)
        self.db2 = (1 / m) * np.sum(self.dZ2, axis=0, keepdims=True)
        
        self.dZ1 = np.dot(self.dZ2, self.W2.T) * self._relu_derivative(self.Z1)
        self.dW1 = (1 / m) * np.dot(X.T, self.dZ1)
        self.db1 = (1 / m) * np.sum(self.dZ1, axis=0, keepdims=True)

    def update_parameters(self, alpha):
        self.W1 -= alpha * self.dW1
        self.b1 -= alpha * self.db1
        self.W2 -= alpha * self.dW2
        self.b2 -= alpha * self.b2

number_of_features = X_train.shape[1]
model = AnomalyDetectorNET(input_dim=number_of_features, hidden_dim=8)

print("Starting AI Training")
for epoch in range(1001):
    predictions = model.forward(X_train)
    loss = model.compute_loss(Y_train, predictions)
    model.backward(X_train, Y_train)
    
    model.dW1 = np.clip(model.dW1, -1.0, 1.0)
    model.db1 = np.clip(model.db1, -1.0, 1.0)
    model.dW2 = np.clip(model.dW2, -1.0, 1.0)
    model.db2 = np.clip(model.db2, -1.0, 1.0)
    
    model.update_parameters(alpha=0.01)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} -> Training Loss: {loss:.6f}")

print("Training finished successfully")

test_predictions = model.forward(X_test)
binary_predictions = (test_predictions > 0.5).astype(float)
test_accuracy = np.mean(binary_predictions == Y_test) * 100
print(f"Final Model Accuracy: {test_accuracy:.2f}%")