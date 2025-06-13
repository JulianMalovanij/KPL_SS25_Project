import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Parameter
store_id = 1
dept_id = 1
forecast_weeks = 104
look_back = 12  # wie viele Wochen zurück das Modell schaut
# -----------------------------

# 📦 Zeitreihe laden
import os

db_path = os.path.join(os.getcwd(), "walmart.db")
conn = sqlite3.connect(db_path)

query = f"""
    SELECT Date, WeeklySales
    FROM WeeklySales
    WHERE StoreID = {store_id} AND DeptID = {dept_id}
    ORDER BY Date
"""
df = pd.read_sql(query, conn, parse_dates=["Date"])
df.set_index("Date", inplace=True)
series = df["WeeklySales"].values.reshape(-1, 1)

# 📊 Skalieren (0-1)
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

# 🧱 Supervised Format (X: Sequenz, y: nächster Wert)
X, y = [], []
for i in range(len(series_scaled) - look_back):
    X.append(series_scaled[i:i + look_back])
    y.append(series_scaled[i + look_back])
X, y = np.array(X), np.array(y)

# 📐 LSTM braucht 3D-Input
X = X.reshape((X.shape[0], X.shape[1], 1))

# 🧠 LSTM-Modell
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, verbose=1)

# 🔮 2 Jahre Forecast
forecast_scaled = []
last_window = series_scaled[-look_back:].copy()

for _ in range(forecast_weeks):
    input_seq = last_window.reshape((1, look_back, 1))
    pred = model.predict(input_seq, verbose=0)[0][0]
    forecast_scaled.append(pred)
    last_window = np.append(last_window[1:], [[pred]], axis=0)

# 🔁 Zurückskalieren
forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

# 📆 Zeitachse
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(weeks=1), periods=forecast_weeks, freq='W')

# 📈 Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["WeeklySales"], label="Historisch", color="black")
plt.plot(future_dates, forecast, label="LSTM Prognose (2 Jahre)", color="orange")
plt.title(f"LSTM Prognose – Store {store_id}, Dept {dept_id}")
plt.xlabel("Datum")
plt.ylabel("Verkäufe")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
