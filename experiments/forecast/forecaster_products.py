# forecaster_products.py
# Prognose der Produktnachfrage pro Lager mit Prophet (basierend auf neuer walmart.db)

import pandas as pd
from prophet import Prophet


def forecast_demand_prophet(df_hist, product_code, warehouse_code):
    df = df_hist.loc[(df_hist["ProductCode"] == product_code) & (df_hist["WarehouseCode"] == warehouse_code),
    ["Date", "OrderDemand"]].sort_values("Date").reset_index(drop=True)

    # Umbenennen für Prophet und Bereinigung der Nachfrage
    df = df.rename(columns={"Date": "ds", "OrderDemand": "y"})
    df['y'] = pd.to_numeric(df['y'].astype(str).str.replace("(", "-", regex=False).str.replace(")", "", regex=False),
                            errors='coerce')
    df = df.dropna()

    if df.empty:
        raise ValueError("Keine gültigen Daten für diese Kombination gefunden.")

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=104, freq='W')
    forecast = model.predict(future)
    return df, forecast
