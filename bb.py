# app_dept_level.py
# Streamlit-Dashboard zur Artikelprognose (aggregiert über alle Stores)

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Unterordner für Forecast-Skripte hinzufügen
sys.path.append(os.path.join(os.getcwd(), "3_advanced_analytics"))
from forecast_dept_level import prophet_forecast, arima_forecast, lstm_forecast, load_aggregated_data

# Sidebar: Auswahl
st.sidebar.title("🔍 Prognose pro Artikel (DeptID)")
dept_id = st.sidebar.selectbox("Wähle Department (Artikel)", list(range(1, 100)))
model = st.sidebar.selectbox("Modell", ["Prophet", "ARIMA", "LSTM"])

# Titel
st.title("📦 Prognose aggregierter Nachfrage pro Artikel")

# Lade Daten
df = load_aggregated_data(dept_id)
st.subheader(f"📈 Historische Verkaufszahlen für Dept {dept_id}")
st.line_chart(df)

# Führe gewähltes Modell aus
st.subheader(f"🔮 Prognose mit {model}")
forecast = None

if model == "Prophet":
    forecast = prophet_forecast(dept_id)
elif model == "ARIMA":
    forecast = arima_forecast(dept_id)
elif model == "LSTM":
    forecast = lstm_forecast(dept_id)

# Visualisierung
if forecast is not None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Sales"], label="Historisch")
    ax.plot(forecast.index, forecast.values, label="Forecast")
    ax.set_title(f"Forecast für Artikel (Dept) {dept_id}")
    ax.legend()
    st.pyplot(fig)
