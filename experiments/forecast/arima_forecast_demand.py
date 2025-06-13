# arima_forecast_demand.py
# ARIMA-Prognose der aggregierten Nachfrage pro Produkt & Lager aus HistoricalDemand

import os
import sqlite3

import pandas as pd
from pmdarima import auto_arima


def arima_forecast_demand(product_code: str, warehouse_code: str):
    # Verbindung zur Datenbank
    db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
    conn = sqlite3.connect(db_path)

    # Daten abfragen
    query = f"""
        SELECT Date, OrderDemand FROM HistoricalDemand
        WHERE ProductCode = '{product_code}' AND WarehouseCode = '{warehouse_code}'
        ORDER BY Date
    """
    df = pd.read_sql(query, conn, parse_dates=['Date'])
    conn.close()

    # Daten vorbereiten
    df = df.rename(columns={"Date": "ds", "OrderDemand": "y"})
    df['y'] = pd.to_numeric(df['y'].astype(str)
                            .str.replace("(", "-", regex=False)
                            .str.replace(")", "", regex=False), errors='coerce')
    df = df.dropna()
    df = df.groupby("ds").sum().reset_index()

    # Modell trainieren
    model = auto_arima(df['y'], seasonal=False, suppress_warnings=True)

    # Prognose für die nächsten 104 Wochen (2 Jahre)
    forecast = model.predict(n_periods=104)
    future_dates = pd.date_range(start=df['ds'].max() + pd.Timedelta(weeks=1), periods=104, freq='W')
    forecast_df = pd.DataFrame({"ds": future_dates, "yhat": forecast})

    return df, forecast_df


# Optional: direktes Ausführen zur Prüfung
if __name__ == "__main__":
    df_hist, df_forecast = arima_forecast_demand("Prod_001", "WH_01")
    print(df_forecast.head())
