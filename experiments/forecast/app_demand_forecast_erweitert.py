# app_demand_forecast.py
# Streamlit-Dashboard für Prophet- und ARIMA-Prognose pro Produkt & Lager, Produkt, Kategorie & Lager, und Kategorie gesamt

import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Pfad zum Ordner 'experiments' hinzufügen
sys.path.append(os.path.join(os.getcwd(), "experiments"))

from experiments.forecast.forecaster_products import forecast_demand_prophet
from experiments.forecast.arima_forecast_demand import arima_forecast_demand

# Titel
st.title("📦 Nachfrageprognose Dashboard")
st.markdown("Prognose der aggregierten Nachfrage nach verschiedenen Dimensionen")


# Verfügbare Kombinationen laden
@st.cache_data
def get_available_combinations():
    db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
    conn = sqlite3.connect(db_path)
    combo_query = """
                  SELECT DISTINCT ProductCode, WarehouseCode, ProductCategory
                  FROM HistoricalDemand
                  ORDER BY ProductCode, WarehouseCode \
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
model_choice = st.radio("🔍 Modell wählen", ["Prophet", "ARIMA"], horizontal=True)

# 1. Prognose nach Produkt & Lager mit Kategorie-Filter
st.header("🔢 Prognose nach Produkt & Lager")
selected_combo = st.selectbox(
    "Produkt & Lager auswählen",
    available_combinations.apply(
        lambda row: f"{row['ProductCategory']} | {row['ProductCode']} | {row['WarehouseCode']}", axis=1)
)

if selected_combo and selected_combo.count("|") == 2:
    selected_category, product_code, warehouse_code = [x.strip() for x in selected_combo.split("|")]

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

# 2. Prognose nach Kategorie & Lager
st.header("🔢 Prognose nach Kategorie & Lager")
selected_cat_lager = st.selectbox(
    "Kategorie & Lager auswählen",
    available_cat_lager.apply(lambda row: f"{row['ProductCategory']} | {row['WarehouseCode']}", axis=1)
)

if selected_cat_lager and selected_cat_lager.count("|") == 1:
    selected_category, warehouse_code = [x.strip() for x in selected_cat_lager.split("|")]

    try:
        db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT Date, OrderDemand FROM HistoricalDemand
            WHERE ProductCategory = '{selected_category}' AND WarehouseCode = '{warehouse_code}'
            ORDER BY Date
        """
        df_combo = pd.read_sql(query, conn, parse_dates=['Date'])
        conn.close()

        df_combo = df_combo.rename(columns={"Date": "ds", "OrderDemand": "y"})
        df_combo['y'] = pd.to_numeric(
            df_combo['y'].astype(str).str.replace("(", "-", regex=False).str.replace(")", "", regex=False),
            errors='coerce')
        df_combo = df_combo.dropna()
        df_combo = df_combo.groupby("ds").sum().reset_index()

        if model_choice == "Prophet":
            from prophet import Prophet

            model = Prophet()
            model.fit(df_combo)
            future = model.make_future_dataframe(periods=104, freq='W')
            forecast_combo = model.predict(future)
            forecast_combo = forecast_combo[['ds', 'yhat']]
        else:
            from pmdarima import auto_arima

            y = df_combo['y'].values
            model = auto_arima(y, seasonal=True, m=52)
            forecast_vals = model.predict(n_periods=104)
            future_dates = pd.date_range(start=df_combo['ds'].max() + pd.Timedelta(weeks=1), periods=104, freq='W')
            forecast_combo = pd.DataFrame({"ds": future_dates, "yhat": forecast_vals})

        st.subheader(f"📊 Prognose für Kategorie {selected_category} im Lager {warehouse_code} ({model_choice})")
        figc, axc = plt.subplots(figsize=(10, 4))
        axc.plot(df_combo['ds'], df_combo['y'], label='Historisch')
        axc.plot(forecast_combo['ds'], forecast_combo['yhat'], label='Prognose')
        axc.set_xlabel("Datum")
        axc.set_ylabel("Nachfrage")
        axc.legend()
        st.pyplot(figc)

    except Exception as e:
        st.error(f"Fehler bei Kategorie-Lager-Vorhersage: {e}")

# 3. Gesamte Nachfrage pro Produkt über alle Lager
st.header("📦 Prognose pro Produkt gesamt")
selected_product = st.selectbox("Produkt auswählen", available_products)

try:
    db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT Date, OrderDemand FROM HistoricalDemand
        WHERE ProductCode = '{selected_product}'
        ORDER BY Date
    """
    df_prod = pd.read_sql(query, conn, parse_dates=['Date'])
    conn.close()

    df_prod = df_prod.rename(columns={"Date": "ds", "OrderDemand": "y"})
    df_prod['y'] = pd.to_numeric(
        df_prod['y'].astype(str).str.replace("(", "-", regex=False).str.replace(")", "", regex=False), errors='coerce')
    df_prod = df_prod.dropna()
    df_prod = df_prod.groupby("ds").sum().reset_index()

    if model_choice == "Prophet":
        from prophet import Prophet

        model = Prophet()
        model.fit(df_prod)
        future = model.make_future_dataframe(periods=104, freq='W')
        forecast_prod = model.predict(future)
        forecast_prod = forecast_prod[['ds', 'yhat']]
    else:
        from pmdarima import auto_arima

        y = df_prod['y'].values
        model = auto_arima(y, seasonal=True, m=52)
        forecast_vals = model.predict(n_periods=104)
        future_dates = pd.date_range(start=df_prod['ds'].max() + pd.Timedelta(weeks=1), periods=104, freq='W')
        forecast_prod = pd.DataFrame({"ds": future_dates, "yhat": forecast_vals})

    st.subheader(f"📈 Prognose für Produkt {selected_product} über alle Lager ({model_choice})")
    figp, axp = plt.subplots(figsize=(10, 4))
    axp.plot(df_prod['ds'], df_prod['y'], label='Historisch')
    axp.plot(forecast_prod['ds'], forecast_prod['yhat'], label='Prognose')
    axp.set_xlabel("Datum")
    axp.set_ylabel("Nachfrage")
    axp.legend()
    st.pyplot(figp)

except Exception as e:
    st.error(f"Fehler bei Produkt-Prognose: {e}")

# 4. Gesamte Nachfrage pro Kategorie über alle Lager
st.header("📦 Prognose pro Kategorie gesamt")
selected_category_total = st.selectbox("Kategorie auswählen", available_categories)

try:
    db_path = os.path.join(os.path.dirname(os.getcwd()), "walmart.db")
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT Date, OrderDemand FROM HistoricalDemand
        WHERE ProductCategory = '{selected_category_total}'
        ORDER BY Date
    """
    df_cat = pd.read_sql(query, conn, parse_dates=['Date'])
    conn.close()

    df_cat = df_cat.rename(columns={"Date": "ds", "OrderDemand": "y"})
    df_cat['y'] = pd.to_numeric(
        df_cat['y'].astype(str).str.replace("(", "-", regex=False).str.replace(")", "", regex=False), errors='coerce')
    df_cat = df_cat.dropna()
    df_cat = df_cat.groupby("ds").sum().reset_index()

    if model_choice == "Prophet":
        from prophet import Prophet

        model = Prophet()
        model.fit(df_cat)
        future = model.make_future_dataframe(periods=104, freq='W')
        forecast_cat = model.predict(future)
        forecast_cat = forecast_cat[['ds', 'yhat']]
    else:
        from pmdarima import auto_arima

        y = df_cat['y'].values
        model = auto_arima(y, seasonal=True, m=52)
        forecast_vals = model.predict(n_periods=104)
        future_dates = pd.date_range(start=df_cat['ds'].max() + pd.Timedelta(weeks=1), periods=104, freq='W')
        forecast_cat = pd.DataFrame({"ds": future_dates, "yhat": forecast_vals})

    st.subheader(f"📈 Prognose für Kategorie {selected_category_total} über alle Lager ({model_choice})")
    figk, axk = plt.subplots(figsize=(10, 4))
    axk.plot(df_cat['ds'], df_cat['y'], label='Historisch')
    axk.plot(forecast_cat['ds'], forecast_cat['yhat'], label='Prognose')
    axk.set_xlabel("Datum")
    axk.set_ylabel("Nachfrage")
    axk.legend()
    st.pyplot(figk)

except Exception as e:
    st.error(f"Fehler bei Kategorie-Prognose: {e}")
