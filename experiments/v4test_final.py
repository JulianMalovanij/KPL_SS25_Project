import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os
import sqlite3
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Zusatzfunktionen für KPIs
@st.cache_data
def calculate_kpis(df):
    total = df['y'].sum()
    weekly_avg = df.set_index('ds').resample('W').sum().mean().values[0]
    volatility = df['y'].std()
    growth = ((df['y'].iloc[-1] - df['y'].iloc[0]) / df['y'].iloc[0]) * 100 if df['y'].iloc[0] != 0 else 0
    return total, weekly_avg, volatility, growth

# Pfad zum Ordner 'experiments' hinzufügen
sys.path.append(os.path.join(os.getcwd(), "experiments"))
from forecast_demand import forecast_demand_prophet
from arima_forecast_demand import arima_forecast_demand

# Titel
st.set_page_config(layout="wide")
st.title("📦 Nachfrageanalyse- & Prognose-Tool")
st.markdown("Erkunden Sie Nachfrageprognosen nach Produkt, Lager und Kategorie mit KPIs, interaktiven Plots und Modellvergleich")

# Verfügbare Kombinationen laden
@st.cache_data
def get_available_combinations():
    db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
    conn = sqlite3.connect(db_path)
    combo_query = """
        SELECT DISTINCT ProductCode, WarehouseCode, ProductCategory 
        FROM HistoricalDemand ORDER BY ProductCode, WarehouseCode
    """
    prod_query = "SELECT DISTINCT ProductCode FROM HistoricalDemand ORDER BY ProductCode"
    cat_query = "SELECT DISTINCT ProductCategory FROM HistoricalDemand ORDER BY ProductCategory"
    cat_lager_query = "SELECT DISTINCT ProductCategory, WarehouseCode FROM HistoricalDemand ORDER BY ProductCategory, WarehouseCode"
    df_combos = pd.read_sql(combo_query, conn)
    df_products = pd.read_sql(prod_query, conn)
    df_categories = pd.read_sql(cat_query, conn)
    df_cat_lager = pd.read_sql(cat_lager_query, conn)
    conn.close()
    return df_combos, df_products['ProductCode'].tolist(), df_categories['ProductCategory'].tolist(), df_cat_lager

available_combinations, available_products, available_categories, available_cat_lager = get_available_combinations()

# Auswahl des Modells
model_choices = st.multiselect("🔍 Modell(e) auswählen", ["Prophet", "ARIMA", "Holt-Winters"], default=["Prophet"])
forecast_period = st.slider("⏱️ Prognosezeitraum (Wochen)", 1, 104, 52)

# Forecast-Funktion Holt-Winters
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

# Forecast-Wrapper
@st.cache_data
def generate_forecasts(df, periods):
    forecasts = {}
    if "Prophet" in model_choices:
        from prophet import Prophet
        prophet_df = df.copy()
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods, freq='W')
        forecast = model.predict(future)[['ds', 'yhat']]
        forecasts['Prophet'] = forecast
    if "ARIMA" in model_choices:
        from pmdarima import auto_arima
        model = auto_arima(df['y'], seasonal=True, m=52)
        pred = model.predict(n_periods=periods)
        future = pd.date_range(df['ds'].max() + pd.Timedelta(weeks=1), periods=periods, freq='W')
        forecasts['ARIMA'] = pd.DataFrame({"ds": future, "yhat": pred})
    if "Holt-Winters" in model_choices:
        forecasts['Holt-Winters'] = holt_winters_forecast(df, periods)
    return forecasts

# Auswahl Produkt und Lager
st.subheader("📦 Prognose nach Produkt & Lager")
selected_combo = st.selectbox("Produkt & Lager auswählen", available_combinations.apply(lambda row: f"{row['ProductCategory']} | {row['ProductCode']} | {row['WarehouseCode']}", axis=1))

if selected_combo and selected_combo.count("|") == 2:
    selected_category, product_code, warehouse_code = [x.strip() for x in selected_combo.split("|")]
    try:
        db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT Date, OrderDemand FROM HistoricalDemand
            WHERE ProductCode = '{product_code}' AND WarehouseCode = '{warehouse_code}'
            ORDER BY Date
        """
        df = pd.read_sql(query, conn, parse_dates=['Date'])
        conn.close()

        df = df.rename(columns={"Date": "ds", "OrderDemand": "y"})
        df['y'] = pd.to_numeric(df['y'].astype(str).str.replace("(", "-", regex=False).str.replace(")", "", regex=False), errors='coerce')
        df = df.dropna().groupby("ds").sum().reset_index()

        total, avg, std, growth = calculate_kpis(df)
        st.metric("📦 Gesamtnachfrage", f"{int(total):,}")
        st.metric("📊 Durchschnitt/Woche", f"{avg:.1f}")
        st.metric("📈 Volatilität", f"{std:.1f}")
        st.metric("📈 Wachstum", f"{growth:.1f}%")

        forecasts = generate_forecasts(df, forecast_period)

        # Interaktives Plotly-Diagramm
        base = px.line(df, x="ds", y="y", title="🔍 Nachfrageprognose", labels={"ds": "Datum", "y": "Nachfrage"})
        for method, forecast_df in forecasts.items():
            base.add_scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name=f"{method}-Forecast")
        st.plotly_chart(base, use_container_width=True)

        # Exportoptionen
        if st.checkbox("📥 Forecast-Daten als CSV exportieren"):
            for method, forecast_df in forecasts.items():
                st.download_button(f"⬇️ Download {method}-Forecast", forecast_df.to_csv(index=False), f"{method}_forecast.csv", "text/csv")

    except Exception as e:
        st.error(f"Fehler: {e}")
