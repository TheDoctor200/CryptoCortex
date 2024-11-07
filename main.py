import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Daten abrufen (Beispiel: Tesla)
ticker = "TSLA"
data = yf.download(ticker, start="2010-01-01", end="2024-01-01")
data = data[['Close']]

# 2. Normalisieren der Daten
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Train-Test-Daten erstellen
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# 4. Aufteilen der Daten in Trainings- und Testset
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. LSTM-Modell erstellen
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Modell trainieren
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 7. Vorhersage für die Testdaten
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_pred = model.predict(X_test)

# 8. Daten zurückskalieren und visualisieren
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform([y_test])

# Visualisierung
plt.plot(y_test[0], label="Echte Kurse")
plt.plot(y_pred[:, 0], label="Vorhergesagte Kurse")
plt.legend()
plt.title(f"Vorhersage für {ticker}")
plt.show()

