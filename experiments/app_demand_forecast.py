# app_demand_forecast.py
# Streamlit-Dashboard für Prophet- und ARIMA-Prognose pro Produkt & Lager und gesamt je Produkt

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import sqlite3

# Pfad zum Ordner 'experiments' hinzufügen
sys.path.append(os.path.join(os.getcwd(), "experiments"))

from forecast_demand import forecast_demand_prophet
from arima_forecast_demand import arima_forecast_demand

# Titel
st.title("📦 Nachfrageprognose Dashboard")
st.markdown("Prognose der aggregierten Nachfrage je Produkt & Lager")

# Verfügbare Kombinationen laden
@st.cache_data
def get_available_combinations():
    db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
    conn = sqlite3.connect(db_path)
    combo_query = "SELECT DISTINCT ProductCode, WarehouseCode FROM HistoricalDemand ORDER BY ProductCode, WarehouseCode"
    prod_query = "SELECT DISTINCT ProductCode FROM HistoricalDemand ORDER BY ProductCode"
    df_combos = pd.read_sql(combo_query, conn)
    df_products = pd.read_sql(prod_query, conn)
    conn.close()
    return df_combos, df_products['ProductCode'].tolist()

available_combinations, available_products = get_available_combinations()

# Auswahl des Modells
model_choice = st.radio("🔍 Modell wählen", ["Prophet", "ARIMA"], horizontal=True)

# 1. Lager-gebundene Auswahl
st.header("🔢 Prognose nach Produkt & Lager")
selected_combo = st.selectbox(
    "Produkt & Lager auswählen",
    available_combinations.apply(lambda row: f"{row['ProductCode']} | {row['WarehouseCode']}", axis=1)
)
product_code, warehouse_code = selected_combo.split(" | ")

if product_code and warehouse_code:
    try:
        if model_choice == "Prophet":
            df, forecast = forecast_demand_prophet(product_code, warehouse_code)
        else:
            df, forecast = arima_forecast_demand(product_code, warehouse_code)

        st.subheader(f"📈 Prognose für Produkt {product_code} im Lager {warehouse_code} ({model_choice})")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['ds'], df['y'], label='Historisch')
        ax.plot(forecast['ds'], forecast['yhat'], label='Prognose')
        ax.set_xlabel("Datum")
        ax.set_ylabel("Nachfrage")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Fehler beim Forecast: {e}")

# 2. Gesamte Nachfrage pro Produkt über alle Lager (mit Modellwahl)
st.header("📦 Gesamtprognose je Produkt (alle Lager)")
selected_product = st.selectbox("Produkt auswählen (gesamt)", available_products)

if selected_product:
    try:
        db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT Date, OrderDemand FROM HistoricalDemand
            WHERE ProductCode = '{selected_product}'
            ORDER BY Date
        """
        df_total = pd.read_sql(query, conn, parse_dates=['Date'])
        conn.close()

        df_total = df_total.rename(columns={"Date": "ds", "OrderDemand": "y"})
        df_total['y'] = pd.to_numeric(df_total['y'].astype(str).str.replace("(", "-", regex=False).str.replace(")", "", regex=False), errors='coerce')
        df_total = df_total.dropna()
        df_total = df_total.groupby("ds").sum().reset_index()

        if model_choice == "Prophet":
            from prophet import Prophet
            model = Prophet()
            model.fit(df_total)
            future = model.make_future_dataframe(periods=104, freq='W')
            forecast_total = model.predict(future)
            forecast_total = forecast_total[['ds', 'yhat']]
        else:
            from pmdarima import auto_arima
            model = auto_arima(df_total['y'], seasonal=False, suppress_warnings=True)
            forecast_vals = model.predict(n_periods=104)
            future_dates = pd.date_range(start=df_total['ds'].max() + pd.Timedelta(weeks=1), periods=104, freq='W')
            forecast_total = pd.DataFrame({"ds": future_dates, "yhat": forecast_vals})

        st.subheader(f"📊 Gesamtprognose für Produkt {selected_product} über alle Lager ({model_choice})")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(df_total['ds'], df_total['y'], label='Historisch')
        ax2.plot(forecast_total['ds'], forecast_total['yhat'], label='Prognose')
        ax2.set_xlabel("Datum")
        ax2.set_ylabel("Nachfrage")
        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Fehler bei Gesamtprognose: {e}")
