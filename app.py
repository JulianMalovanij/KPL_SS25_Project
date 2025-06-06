import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

# 🔧 Unterordner mit Forecast-Skripten hinzufügen
sys.path.append(os.path.join(os.getcwd(), "3_advanced_analytics"))

# ----------------------------
# Einstellungen
# ----------------------------
db_path = "walmart.db"
data_dir = "C:/KIProjekt"
os.chdir(data_dir)

# ----------------------------
# Auswahlfelder in Sidebar
# ----------------------------
st.sidebar.title("🔍 Auswahl")
store_id = st.sidebar.selectbox("Store", list(range(1, 11)))
dept_id = st.sidebar.selectbox("Department", list(range(1, 100)))
model_option = st.sidebar.selectbox("Modell", ["Prophet", "ARIMA", "LSTM"])

# ----------------------------
# Daten aus DB laden
# ----------------------------
def load_timeseries(store, dept):
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT Date, WeeklySales FROM WeeklySales
        WHERE StoreID = {store} AND DeptID = {dept}
        ORDER BY Date
    """
    df = pd.read_sql(query, conn, parse_dates=['Date'])
    conn.close()
    return df.set_index("Date")

# ----------------------------
# Forecasts importieren
# ----------------------------
def run_prophet_forecast(store, dept):
    from forecast_prophet import prophet_forecast
    return prophet_forecast(store, dept)

def run_arima_forecast(store, dept):
    from arima_forecast import arima_forecast
    return arima_forecast(store, dept)

def run_lstm_forecast(store, dept):
    from lstm_forecast import lstm_forecast
    return lstm_forecast(store, dept)

# ----------------------------
# Visualisierung Forecasts
# ----------------------------
st.title("📊 Verkaufsprognosen & Analysen")

# Rohdaten anzeigen
st.subheader("🗂️ Verkaufszeitreihe")
df = load_timeseries(store_id, dept_id)
st.line_chart(df)

# Forecast anzeigen
st.subheader(f"🔮 Prognose mit {model_option}")
if model_option == "Prophet":
    forecast = run_prophet_forecast(store_id, dept_id)
elif model_option == "ARIMA":
    forecast = run_arima_forecast(store_id, dept_id)
elif model_option == "LSTM":
    forecast = run_lstm_forecast(store_id, dept_id)

# Forecast plotten
if forecast is not None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["WeeklySales"], label="Historisch")
    ax.plot(forecast.index, forecast.values, label="Forecast")
    ax.set_title(f"Verkäufe Store {store_id} / Dept {dept_id}")
    ax.legend()
    st.pyplot(fig)

# ----------------------------
# Deskriptive Analyse einbinden
# ----------------------------
st.subheader("📈 Deskriptive Analyse")
from descriptive_analyses import run_descriptive_analysis
run_descriptive_analysis(store_id, dept_id)

# ----------------------------
# Zusatz-Visuals (optional)
# ----------------------------
st.subheader("🧪 Zusatz-Visualisierungen")
from visualizations import run_additional_charts
run_additional_charts(store_id, dept_id)
