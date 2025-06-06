import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sqlite3

st.title("📈 Nachfrageprognose mit Prophet")
st.subheader("Interaktive Visualisierung pro Store und Abteilung")

# Datenbankverbindung
conn = sqlite3.connect("../walmart.db")

# Alle verfügbaren Kombinationen abrufen
combos = pd.read_sql("SELECT DISTINCT StoreID, DeptID FROM WeeklySales ORDER BY StoreID, DeptID;", conn)

# Auswahlfelder
store_id = st.selectbox("Wähle Store", sorted(combos["StoreID"].unique()))
dept_id = st.selectbox("Wähle Department", sorted(combos[combos["StoreID"] == store_id]["DeptID"].unique()))

# Dateiname für Prognose
filename = f"forecast_store{store_id}_dept{dept_id}.csv"
filepath = os.path.join("forecast_results", filename)

# Lade Prognose, falls Datei vorhanden
if os.path.exists(filepath):
    forecast = pd.read_csv(filepath, parse_dates=["ds"])
    query = f"""
        SELECT Date, WeeklySales
        FROM WeeklySales
        WHERE StoreID = {store_id} AND DeptID = {dept_id}
        ORDER BY Date
    """
    history = pd.read_sql(query, conn, parse_dates=["Date"])

    # Plot mit Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["Date"], history["WeeklySales"], label="Historisch", color="black")
    ax.plot(forecast["ds"], forecast["yhat"], label="Prognose", color="blue")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                    color="blue", alpha=0.2, label="Konfidenzintervall")

    ax.set_title(f"Store {store_id} – Dept {dept_id}")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Verkäufe")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
else:
    st.warning("⚠️ Für diese Kombination liegt keine Prognosedatei vor.")
