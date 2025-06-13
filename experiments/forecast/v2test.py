import os
import sqlite3
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from prophet import Prophet
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


# Holt-Winters Forecast (statt ARIMA)
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


# Streamlit UI
st.title("📊 Walmart Forecast mit Holt-Winters")

df = load_sales_data()
store_ids = df["StoreID"].dropna().unique()
store_id = st.selectbox("Wähle eine Store-ID", sorted(store_ids))

forecast_method = st.radio("Forecast-Methode", ["Prophet", "Holt-Winters"])
forecast_periods = st.slider("Prognosezeitraum (Wochen)", 1, 52, 12)

# Wöchentliche Verkaufsdaten pro Store aggregieren
store_df = df[df["StoreID"] == store_id]
store_df = store_df.groupby("Date").agg({"WeeklySales": "sum"}).reset_index()
store_df = store_df.rename(columns={"WeeklySales": "y"})

# Forecast anwenden
if forecast_method == "Prophet":
    forecast_df = prophet_forecast(store_df, forecast_periods)
else:
    forecast_df = holt_winters_forecast(store_df, forecast_periods)

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
