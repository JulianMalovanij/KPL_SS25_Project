import os
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd

# Welche Prognose soll visualisiert werden?
store_id = 1
dept_id = 1

# Pfad zur CSV-Datei mit der Vorhersage
filename = f"forecast_store{store_id}_dept{dept_id}.csv"
forecast_path = os.path.join("../../3_advanced_analytics/forecast_results", filename)

# Prognosedaten laden
forecast_df = pd.read_csv(forecast_path, parse_dates=["ds"])

# Optional: historische Werte aus der Datenbank laden
conn = sqlite3.connect("../walmart.db")
query = f"""
    SELECT Date, WeeklySales
    FROM WeeklySales
    WHERE StoreID = {store_id} AND DeptID = {dept_id}
    ORDER BY Date
"""
history_df = pd.read_sql(query, conn, parse_dates=["Date"])

# Plot erstellen
plt.figure(figsize=(12, 6))
plt.plot(history_df["Date"], history_df["WeeklySales"], label="Historisch", color="black")
plt.plot(forecast_df["ds"], forecast_df["yhat"], label="Prognose", color="blue")
plt.fill_between(forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"],
                 color="blue", alpha=0.2, label="Konfidenzintervall")

plt.title(f"Prognose für Store {store_id}, Dept {dept_id}")
plt.xlabel("Datum")
plt.ylabel("Verkäufe")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
