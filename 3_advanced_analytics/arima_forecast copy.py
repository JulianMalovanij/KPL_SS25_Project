print("Script gestartet!")



import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Parameter: Kombination aus Store und Department
store_id = 1
dept_id = 1

# Verbindung zur Datenbank
conn = sqlite3.connect("../walmart.db")
query = f"""
    SELECT Date, WeeklySales
    FROM WeeklySales
    WHERE StoreID = {store_id} AND DeptID = {dept_id}
    ORDER BY Date
"""
df = pd.read_sql(query, conn, parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Nur Zeitreihe extrahieren
series = df["WeeklySales"]

# Train/Test-Split (z. B. letzte 12 Wochen als Test)
train = series[:-12]
test = series[-12:]

# Einfaches ARIMA-Modell (Parameter: p=2, d=1, q=2)
model = ARIMA(train, order=(2,1,2))
fitted_model = model.fit()

# Prognose für 12 Wochen
forecast = fitted_model.forecast(steps=12)

# Bewertung (RMSE)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"RMSE: {rmse:.2f}")

# Plot: Original + Prognose
plt.figure(figsize=(10,5))
plt.plot(series.index, series.values, label="Original", color="black")
plt.plot(test.index, forecast, label="ARIMA Forecast", color="red")
plt.title(f"ARIMA Prognose – Store {store_id}, Dept {dept_id}")
plt.xlabel("Datum")
plt.ylabel("Verkäufe")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
