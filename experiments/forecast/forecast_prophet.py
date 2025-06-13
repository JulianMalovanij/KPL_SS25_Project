# Verbindung zur SQLite-Datenbank
import os
import sqlite3

import pandas as pd
from prophet import Prophet

db_path = os.path.join(os.getcwd(), "walmart.db")
conn = sqlite3.connect(db_path)

# Ausgabeordner für die Prognose-Ergebnisse
output_dir = "../../3_advanced_analytics/forecast_results"
os.makedirs(output_dir, exist_ok=True)

# Alle Store/Dept-Kombinationen laden
combinations = pd.read_sql(
    "SELECT DISTINCT StoreID, DeptID FROM WeeklySales ORDER BY StoreID, DeptID;",
    conn
)

# Für jede Kombination ein Prophet-Modell trainieren
for _, row in combinations.iterrows():
    store_id = row["StoreID"]
    dept_id = row["DeptID"]

    # Verkaufszeitreihe aus der Datenbank holen
    query = f'''
        SELECT Date, WeeklySales
        FROM WeeklySales
        WHERE StoreID = {store_id} AND DeptID = {dept_id}
        ORDER BY Date
    '''
    df = pd.read_sql(query, conn)

    # Prophet benötigt Spalten "ds" und "y"
    df.rename(columns={"Date": "ds", "WeeklySales": "y"}, inplace=True)

    # Kombinationen mit zu wenig Daten überspringen
    if len(df) < 20:
        continue

    # Modell trainieren
    model = Prophet()
    model.fit(df)

    # Zukunftsdaten für 12 Wochen erstellen
    future = model.make_future_dataframe(periods=12, freq='W')
    forecast = model.predict(future)

    # Ergebnis als CSV speichern
    filename = f"forecast_store{store_id}_dept{dept_id}.csv"
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
        os.path.join(output_dir, filename), index=False
    )

print("✅ Alle Prognosen wurden berechnet und gespeichert.")
