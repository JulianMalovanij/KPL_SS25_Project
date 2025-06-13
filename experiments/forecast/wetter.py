import os
import sqlite3
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

# Titel
st.set_page_config(layout="wide")
st.title("🌡️ Wetterbasierte Verkaufsprognose")
st.markdown(
    "Dieses Dashboard nutzt Temperatur, Feiertage und weitere Store-Features zur verbesserten Prognose der Weekly Sales.")


# Daten laden
@st.cache_data
def load_weather_sales_data():
    db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
    conn = sqlite3.connect(db_path)
    sales = pd.read_sql("SELECT Date, WeeklySales, StoreID FROM WeeklySales", conn)
    features = pd.read_sql(
        "SELECT Date, StoreID, Temperature, FuelPrice, CPI, Unemployment, IsHoliday FROM StoreFeature", conn)
    conn.close()

    sales['Date'] = pd.to_datetime(sales['Date'])
    features['Date'] = pd.to_datetime(features['Date'])

    df = pd.merge(sales, features, on=["Date", "StoreID"], how="inner")
    df = df.dropna()
    df['WeeklySales'] = pd.to_numeric(df['WeeklySales'], errors='coerce')
    df = df.dropna()
    return df


# KPIs
@st.cache_data
def calculate_kpis(df):
    total = df['y'].sum()
    weekly_avg = df.set_index('ds').resample('W').sum().mean().values[0]
    volatility = df['y'].std()
    growth = ((df['y'].iloc[-1] - df['y'].iloc[0]) / df['y'].iloc[0]) * 100 if df['y'].iloc[0] != 0 else 0
    return total, weekly_avg, volatility, growth


# Forecast mit Prophet
def prophet_with_weather(df, periods, regressors):
    df = df.rename(columns={"Date": "ds", "WeeklySales": "y"})
    df = df.sort_values("ds")
    df = df[["ds", "y"] + regressors]
    model = Prophet()
    for reg in regressors:
        model.add_regressor(reg)
    model.fit(df)

    future = df[["ds"] + regressors].copy()
    last_date = df["ds"].max()
    for i in range(1, periods + 1):
        next_date = last_date + timedelta(weeks=i)
        new_row = {"ds": next_date}
        for reg in regressors:
            new_row[reg] = df[reg].iloc[-1]  # zuletzt bekannte Werte einfrieren
        future = pd.concat([future, pd.DataFrame([new_row])], ignore_index=True)

    forecast = model.predict(future)
    forecast_future = forecast[forecast["ds"] > df["ds"].max()]
    return df, forecast_future[["ds", "yhat"]]


# Auswahl
df_raw = load_weather_sales_data()
store_ids = sorted(df_raw['StoreID'].unique())
selected_store = st.selectbox("🏬 Store auswählen", store_ids)
forecast_period = st.slider("⏱️ Prognosezeitraum (Wochen)", 1, 52, 12)

regressor_options = ["Temperature", "FuelPrice", "CPI", "Unemployment", "IsHoliday"]
selected_regressors = st.multiselect("🧮 Externe Einflussfaktoren", regressor_options,
                                     default=["Temperature", "FuelPrice", "IsHoliday"])

# Daten vorbereiten
store_df = df_raw[df_raw['StoreID'] == selected_store].copy()
train_df, forecast_df = prophet_with_weather(store_df, forecast_period, selected_regressors)

# KPIs anzeigen
kpi_df = train_df.rename(columns={"ds": "ds", "y": "y"})
total, avg, std, growth = calculate_kpis(kpi_df)
st.metric("📦 Gesamtumsatz", f"{int(total):,}")
st.metric("📊 Durchschnitt/Woche", f"{avg:.1f}")
st.metric("📈 Volatilität", f"{std:.1f}")
st.metric("📈 Wachstum", f"{growth:.1f}%")

# Korrelationsmatrix
if st.checkbox("📊 Korrelation anzeigen"):
    corr_df = store_df[["WeeklySales"] + selected_regressors].copy()
    corr = corr_df.corr()
    st.dataframe(corr.style.background_gradient(cmap='RdBu', axis=None))

# Klarer Plot mit Forecast
fig = go.Figure()
fig.add_trace(go.Scatter(x=store_df['Date'], y=store_df['WeeklySales'], mode='lines', name='Historische Verkäufe'))
fig.add_trace(
    go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast', line=dict(dash='dash')))
fig.update_layout(title=f"📉 Wetterbasierte Prognose für Store {selected_store}", xaxis_title="Datum",
                  yaxis_title="Weekly Sales", legend_title="Daten")
st.plotly_chart(fig, use_container_width=True)

# CSV-Export
if st.checkbox("📥 Forecast-Daten als CSV exportieren"):
    st.download_button("⬇️ Download Forecast", forecast_df.to_csv(index=False), "weather_forecast.csv", "text/csv")
