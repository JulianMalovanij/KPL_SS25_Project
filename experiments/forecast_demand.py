# forecast_demand.py
# Prognose der Produktnachfrage pro Lager mit Prophet (basierend auf neuer walmart.db)

import pandas as pd
import sqlite3
from prophet import Prophet
import matplotlib.pyplot as plt

DB_PATH = "../walmart.db"

def load_demand_data(product_code, warehouse_code):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT Date, OrderDemand
        FROM HistoricalDemand
        WHERE ProductCode = '{product_code}' AND WarehouseCode = '{warehouse_code}'
        ORDER BY Date
    """
    df = pd.read_sql(query, conn, parse_dates=['Date'])
    conn.close()

    # Umbenennen für Prophet und Bereinigung der Nachfrage
    df = df.rename(columns={"Date": "ds", "OrderDemand": "y"})
    df['y'] = pd.to_numeric(df['y'].astype(str).str.replace("(", "-", regex=False).str.replace(")", "", regex=False), errors='coerce')
    df = df.dropna()

    return df

def forecast_demand_prophet(product_code, warehouse_code):
    df = load_demand_data(product_code, warehouse_code)
    if df.empty:
        raise ValueError("Keine gültigen Daten für diese Kombination gefunden.")

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=104, freq='W')
    forecast = model.predict(future)
    return df, forecast

def plot_forecast(df, forecast, product_code, warehouse_code):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['ds'], df['y'], label='Historisch')
    ax.plot(forecast['ds'], forecast['yhat'], label='Prognose')
    ax.set_title(f"Nachfrageprognose für Produkt {product_code} im Lager {warehouse_code}")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Nachfrage")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    product = input("🔢 Produktcode eingeben: ")
    warehouse = input("🏬 Lagercode eingeben: ")
    try:
        df, forecast = forecast_demand_prophet(product, warehouse)
        plot_forecast(df, forecast, product, warehouse)
    except ValueError as e:
        print(f"Fehler: {e}")
