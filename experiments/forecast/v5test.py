import os
import sqlite3
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Zusatzfunktionen für KPIs
@st.cache_data
def calculate_kpis(df):
    total = df['y'].sum()
    weekly_avg = df.set_index('ds').resample('W').sum().mean().values[0]
    volatility = df['y'].std()
    growth = ((df['y'].iloc[-1] - df['y'].iloc[0]) / df['y'].iloc[0]) * 100 if df['y'].iloc[0] != 0 else 0
    return total, weekly_avg, volatility, growth


# Titel & Layout
st.set_page_config(layout="wide")
st.title("📈 Verkaufsprognose-Tool (Weekly Sales)")
st.markdown("Analysiere historische Verkaufsdaten mit KPIs, Prognosemodellen und interaktiven Diagrammen")


# Daten laden
@st.cache_data
def load_sales_data():
    db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT Date, WeeklySales, StoreID FROM WeeklySales", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna()
    df['WeeklySales'] = pd.to_numeric(df['WeeklySales'], errors='coerce')
    df = df.dropna()
    return df


# Forecast-Funktionen
def prophet_forecast(df, periods):
    df = df.rename(columns={"ds": "ds", "y": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]


def holt_winters_forecast(df, periods):
    df = df.set_index("ds")
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


def arima_forecast(df, periods):
    model = auto_arima(df['y'], seasonal=True, m=52)
    pred = model.predict(n_periods=periods)
    future = pd.date_range(df['ds'].max() + pd.Timedelta(weeks=1), periods=periods, freq='W')
    return pd.DataFrame({"ds": future, "yhat": pred})


# Forecast-Wrapper
@st.cache_data
def generate_forecasts(df, periods, model_choices):
    forecasts = {}
    if "Prophet" in model_choices:
        forecasts['Prophet'] = prophet_forecast(df.copy(), periods)
    if "ARIMA" in model_choices:
        forecasts['ARIMA'] = arima_forecast(df.copy(), periods)
    if "Holt-Winters" in model_choices:
        forecasts['Holt-Winters'] = holt_winters_forecast(df.copy(), periods)
    return forecasts


# UI Sidebar
with st.sidebar:
    st.header("🧭 Einstellungen")
    df_raw = load_sales_data()
    store_ids = sorted(df_raw['StoreID'].unique())
    selected_store = st.selectbox("🏬 Store auswählen", store_ids)
    model_choices = st.multiselect("📊 Modell(e) auswählen", ["Prophet", "ARIMA", "Holt-Winters"], default=["Prophet"])
    forecast_period = st.slider("📅 Prognosezeitraum (Wochen)", 1, 52, 12)
    show_table = st.checkbox("📋 Rohdaten anzeigen", value=False)
    show_forecast_table = st.checkbox("📈 Forecast-Tabelle anzeigen", value=False)

# Daten aggregieren
store_df = df_raw[df_raw['StoreID'] == selected_store].copy()
store_df = store_df.groupby("Date").agg({"WeeklySales": "sum"}).reset_index()
store_df = store_df.rename(columns={"Date": "ds", "WeeklySales": "y"})

# KPIs anzeigen
total, avg, std, growth = calculate_kpis(store_df)
st.metric("📦 Gesamtumsatz", f"{int(total):,}")
st.metric("📊 Durchschnitt/Woche", f"{avg:.1f}")
st.metric("📈 Volatilität", f"{std:.1f}")
st.metric("📈 Wachstum", f"{growth:.1f}%")

# Prognosen generieren
forecasts = generate_forecasts(store_df, forecast_period, model_choices)

# Interaktives Diagramm
base = px.line(store_df, x="ds", y="y", title=f"🔍 Verkaufsprognose für Store {selected_store}",
               labels={"ds": "Datum", "y": "Verkäufe"})
for method, forecast_df in forecasts.items():
    base.add_scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name=f"{method}-Forecast")
st.plotly_chart(base, use_container_width=True)

# Tabellen anzeigen
if show_table:
    st.subheader("🗃️ Historische Verkaufsdaten")
    st.dataframe(store_df)

if show_forecast_table:
    st.subheader("📈 Prognosewerte")
    for method, forecast_df in forecasts.items():
        st.markdown(f"**{method}-Forecast**")
        st.dataframe(forecast_df)

# Exportoptionen
if st.checkbox("📥 Forecast-Daten als CSV exportieren"):
    for method, forecast_df in forecasts.items():
        st.download_button(f"⬇️ Download {method}-Forecast", forecast_df.to_csv(index=False), f"{method}_forecast.csv",
                           "text/csv")

# Footer
st.markdown("""
---
Erstellt mit ❤️ in Streamlit – Vergleich der Forecast-Modelle: Prophet, Holt-Winters, ARIMA
""")
