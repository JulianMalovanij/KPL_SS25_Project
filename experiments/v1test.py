import streamlit as st
import sqlite3
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import matplotlib.pyplot as plt
import os

# 📁 Relativer Pfad zur Datenbank
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "walmart.db"))

@st.cache_data
def load_sales_data():
    conn = sqlite3.connect(DB_PATH)
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

# ARIMA Forecast
def arima_forecast(df, periods):
    model = ARIMA(df['y'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    last_date = df['Date'].max()
    future_dates = [last_date + timedelta(weeks=i) for i in range(1, periods + 1)]
    return pd.DataFrame({"ds": future_dates, "yhat": forecast})

# Streamlit UI
st.title("📈 Walmart Forecast – Wöchentliche Verkäufe")

df = load_sales_data()
store_ids = df["StoreID"].dropna().unique()
store_id = st.selectbox("Wähle eine Store-ID", sorted(store_ids))

forecast_method = st.radio("Forecast-Methode", ["Prophet", "ARIMA"])
forecast_periods = st.slider("Prognosezeitraum (Wochen)", 1, 52, 12)

# Aggregation: Verkäufe pro Woche und Store (über alle Departments hinweg)
store_df = df[df["StoreID"] == store_id]
store_df = store_df.groupby("Date").agg({"WeeklySales": "sum"}).reset_index()
store_df = store_df.rename(columns={"WeeklySales": "y"})

# Forecast
if forecast_method == "Prophet":
    forecast_df = prophet_forecast(store_df, forecast_periods)
else:
    forecast_df = arima_forecast(store_df, forecast_periods)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(store_df["Date"], store_df["y"], label="Historische Verkäufe", linewidth=2)
ax.plot(forecast_df["ds"], forecast_df["yhat"], label=f"{forecast_method}-Forecast", linestyle="--")
ax.set_title(f"Wöchentliche Verkäufe – Store {store_id}")
ax.set_xlabel("Datum")
ax.set_ylabel("Verkäufe")
ax.legend()
ax.grid(True)

st.pyplot(fig)
