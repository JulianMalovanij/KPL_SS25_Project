import os
import sqlite3
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Relativer Pfad zur Datenbank
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "walmart.db"))


@st.cache_data
def load_sales_data():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM WeeklySales", conn)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "WeeklySales", "StoreID"])
    df["WeeklySales"] = pd.to_numeric(df["WeeklySales"], errors="coerce")
    df = df.dropna(subset=["WeeklySales"])
    return df


# Prophet Forecast
def prophet_forecast(df, periods):
    df = df.rename(columns={"Date": "ds", "y": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]


# Holt-Winters Forecast
def holt_winters_forecast(df, periods):
    df = df.set_index("Date")
    model = ExponentialSmoothing(
        df['y'],
        trend='additive',
        seasonal='additive',
        seasonal_periods=20,
        damped_trend=True
    ).fit()
    forecast = model.forecast(periods)
    future_dates = [df.index.max() + timedelta(weeks=i) for i in range(1, periods + 1)]
    return pd.DataFrame({"ds": future_dates, "yhat": forecast.values})


# ARIMA Forecast
def arima_forecast(df, periods):
    model = ARIMA(df['y'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    last_date = df['Date'].max()
    future_dates = [last_date + timedelta(weeks=i) for i in range(1, periods + 1)]
    return pd.DataFrame({"ds": future_dates, "yhat": forecast})


# Streamlit UI
st.set_page_config(page_title="Walmart Forecast Dashboard", layout="wide")
st.title("📊 Walmart Forecast Dashboard")

with st.sidebar:
    st.header("🧭 Einstellungen")
    df = load_sales_data()
    store_ids = df["StoreID"].dropna().unique()
    store_id = st.selectbox("📍 Wähle eine Store-ID", sorted(store_ids))
    forecast_method = st.radio("🔮 Forecast-Methode", ["Prophet", "Holt-Winters", "ARIMA"])
    forecast_periods = st.slider("⏱️ Prognosezeitraum (Wochen)", 1, 52, 12)
    show_table = st.checkbox("📋 Rohdaten anzeigen", value=False)
    show_forecast_table = st.checkbox("📈 Forecast-Tabelle anzeigen", value=False)

# Wöchentliche Verkaufsdaten pro Store aggregieren
store_df = df[df["StoreID"] == store_id]
store_df = store_df.groupby("Date").agg({"WeeklySales": "sum"}).reset_index()
store_df = store_df.rename(columns={"WeeklySales": "y"})

# Forecast anwenden
if forecast_method == "Prophet":
    forecast_df = prophet_forecast(store_df, forecast_periods)
elif forecast_method == "Holt-Winters":
    forecast_df = holt_winters_forecast(store_df, forecast_periods)
elif forecast_method == "ARIMA":
    forecast_df = arima_forecast(store_df, forecast_periods)

# Visualisierung
st.subheader(f"📆 Prognose für Store {store_id} – Methode: {forecast_method}")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(store_df["Date"], store_df["y"], label="Historische Verkäufe", linewidth=2)
ax.plot(forecast_df["ds"], forecast_df["yhat"], label=f"{forecast_method}-Forecast", linestyle="--")
ax.set_xlabel("Datum")
ax.set_ylabel("Verkäufe")
ax.set_title(f"Wöchentliche Verkäufe – Store {store_id}")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Optionale Tabellenanzeige
if show_table:
    st.subheader("🗃️ Historische Verkaufsdaten")
    st.dataframe(store_df)

if show_forecast_table:
    st.subheader("📈 Prognosewerte")
    st.dataframe(forecast_df)

# Footer
st.markdown("""
---
Erstellt mit ❤️ mit Streamlit – Prognosevergleich: Prophet, Holt-Winters, ARIMA
""")
