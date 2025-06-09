import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
st.markdown("Erkunden Sie Nachfrageprognosen nach Produkt, Lager und Kategorie mit KPIs und Modellvergleich")

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
model_choice = st.radio("🔍 Modell wählen", ["Prophet", "ARIMA", "Holt-Winters"], horizontal=True)

# Container für 4 Prognosearten
cols = st.columns(2)

# Helper-Funktion Holt-Winters Forecast

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

# 1. Prognose nach Produkt & Lager
with cols[0]:
    st.subheader("🔢 Prognose: Produkt & Lager")
    selected_combo = st.selectbox("Produkt & Lager auswählen", available_combinations.apply(lambda row: f"{row['ProductCategory']} | {row['ProductCode']} | {row['WarehouseCode']}", axis=1))

    if selected_combo and selected_combo.count("|") == 2:
        selected_category, product_code, warehouse_code = [x.strip() for x in selected_combo.split("|")]
        try:
            if model_choice == "Prophet":
                df, forecast = forecast_demand_prophet(product_code, warehouse_code)
            elif model_choice == "ARIMA":
                df, forecast = arima_forecast_demand(product_code, warehouse_code)
            else:
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
                forecast = holt_winters_forecast(df, 104)

            total, avg, std, growth = calculate_kpis(df)
            st.metric("📦 Gesamtnachfrage", f"{int(total):,}")
            st.metric("📊 Durchschnitt/Woche", f"{avg:.1f}")
            st.metric("📈 Volatilität", f"{std:.1f}")
            st.metric("📈 Wachstum", f"{growth:.1f}%")

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(df['ds'], df['y'], label='Historisch')
            ax.plot(forecast['ds'], forecast['yhat'], label='Prognose')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Fehler: {e}")

# 2. Prognose Kategorie & Lager
with cols[1]:
    st.subheader("🏷️ Prognose: Kategorie & Lager")
    selected_cat_lager = st.selectbox("Kategorie & Lager auswählen", available_cat_lager.apply(lambda row: f"{row['ProductCategory']} | {row['WarehouseCode']}", axis=1))

    if selected_cat_lager:
        selected_category, warehouse_code = [x.strip() for x in selected_cat_lager.split("|")]
        try:
            db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
            conn = sqlite3.connect(db_path)
            query = f"""
                SELECT Date, OrderDemand FROM HistoricalDemand
                WHERE ProductCategory = '{selected_category}' AND WarehouseCode = '{warehouse_code}'
                ORDER BY Date
            """
            df = pd.read_sql(query, conn, parse_dates=['Date'])
            conn.close()

            df = df.rename(columns={"Date": "ds", "OrderDemand": "y"})
            df['y'] = pd.to_numeric(df['y'].astype(str).str.replace("(", "-", regex=False).str.replace(")", "", regex=False), errors='coerce')
            df = df.dropna().groupby("ds").sum().reset_index()

            if model_choice == "Prophet":
                from prophet import Prophet
                model = Prophet()
                model.fit(df)
                future = model.make_future_dataframe(periods=104, freq='W')
                forecast = model.predict(future)[['ds', 'yhat']]
            elif model_choice == "ARIMA":
                from pmdarima import auto_arima
                model = auto_arima(df['y'], seasonal=True, m=52)
                pred = model.predict(n_periods=104)
                future = pd.date_range(df['ds'].max() + pd.Timedelta(weeks=1), periods=104, freq='W')
                forecast = pd.DataFrame({"ds": future, "yhat": pred})
            else:
                forecast = holt_winters_forecast(df, 104)

            total, avg, std, growth = calculate_kpis(df)
            st.metric("📦 Gesamtnachfrage", f"{int(total):,}")
            st.metric("📊 Durchschnitt/Woche", f"{avg:.1f}")
            st.metric("📈 Volatilität", f"{std:.1f}")
            st.metric("📈 Wachstum", f"{growth:.1f}%")

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(df['ds'], df['y'], label='Historisch')
            ax.plot(forecast['ds'], forecast['yhat'], label='Prognose')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Fehler: {e}")

# 3. Produkt gesamt
st.subheader("📦 Prognose: Produkt gesamt")
selected_product = st.selectbox("Produkt wählen", available_products)
try:
    db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT Date, OrderDemand FROM HistoricalDemand
        WHERE ProductCode = '{selected_product}' ORDER BY Date
    """
    df = pd.read_sql(query, conn, parse_dates=['Date'])
    conn.close()

    df = df.rename(columns={"Date": "ds", "OrderDemand": "y"})
    df['y'] = pd.to_numeric(df['y'].astype(str).str.replace("(", "-", regex=False).str.replace(")", "", regex=False), errors='coerce')
    df = df.dropna().groupby("ds").sum().reset_index()

    if model_choice == "Prophet":
        from prophet import Prophet
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=104, freq='W')
        forecast = model.predict(future)[['ds', 'yhat']]
    elif model_choice == "ARIMA":
        from pmdarima import auto_arima
        model = auto_arima(df['y'], seasonal=True, m=52)
        pred = model.predict(n_periods=104)
        future = pd.date_range(df['ds'].max() + pd.Timedelta(weeks=1), periods=104, freq='W')
        forecast = pd.DataFrame({"ds": future, "yhat": pred})
    else:
        forecast = holt_winters_forecast(df, 104)

    total, avg, std, growth = calculate_kpis(df)
    st.metric("📦 Gesamtnachfrage", f"{int(total):,}")
    st.metric("📊 Durchschnitt/Woche", f"{avg:.1f}")
    st.metric("📈 Volatilität", f"{std:.1f}")
    st.metric("📈 Wachstum", f"{growth:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df['ds'], df['y'], label='Historisch')
    ax.plot(forecast['ds'], forecast['yhat'], label='Prognose')
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.error(f"Fehler: {e}")

# 4. Kategorie gesamt
st.subheader("🏷️ Prognose: Kategorie gesamt")
selected_category = st.selectbox("Kategorie wählen", available_categories)
try:
    db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT Date, OrderDemand FROM HistoricalDemand
        WHERE ProductCategory = '{selected_category}' ORDER BY Date
    """
    df = pd.read_sql(query, conn, parse_dates=['Date'])
    conn.close()

    df = df.rename(columns={"Date": "ds", "OrderDemand": "y"})
    df['y'] = pd.to_numeric(df['y'].astype(str).str.replace("(", "-", regex=False).str.replace(")", "", regex=False), errors='coerce')
    df = df.dropna().groupby("ds").sum().reset_index()

    if model_choice == "Prophet":
        from prophet import Prophet
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=104, freq='W')
        forecast = model.predict(future)[['ds', 'yhat']]
    elif model_choice == "ARIMA":
        from pmdarima import auto_arima
        model = auto_arima(df['y'], seasonal=True, m=52)
        pred = model.predict(n_periods=104)
        future = pd.date_range(df['ds'].max() + pd.Timedelta(weeks=1), periods=104, freq='W')
        forecast = pd.DataFrame({"ds": future, "yhat": pred})
    else:
        forecast = holt_winters_forecast(df, 104)

    total, avg, std, growth = calculate_kpis(df)
    st.metric("📦 Gesamtnachfrage", f"{int(total):,}")
    st.metric("📊 Durchschnitt/Woche", f"{avg:.1f}")
    st.metric("📈 Volatilität", f"{std:.1f}")
    st.metric("📈 Wachstum", f"{growth:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df['ds'], df['y'], label='Historisch')
    ax.plot(forecast['ds'], forecast['yhat'], label='Prognose')
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.error(f"Fehler: {e}")
