import streamlit as st

from layout import with_layout


@with_layout("👋 Willkommen zur Verkaufsanalyse-App")
def page():
    st.write(
        "Diese App bietet drei Hauptfunktionen, die helfen, Verkaufsdaten besser zu verstehen und zu optimieren.")

    st.header("📊 Deskriptive Analyse")
    st.write("""
        - 🔍 Untersuche historische Verkaufsdaten.
        - 📈 Visualisiere Trends, Saisonalitäten und Ausreißer.
        - 🧩 Finde Zusammenhänge.
        """)

    st.divider()

    st.header("📅 Verkaufsvorhersage")
    st.write("""
        - 🤖 Erstelle Verkaufsprognosen mit verschiedenen Modellen wie **Prophet**, **ARIMA** oder **LSTM**.
        - 🗓️ Nutze die Vorhersagen zur Planung und Bestandsoptimierung.
        - 📊 Vergleiche die Modellgenauigkeiten.
        """)

    st.divider()

    st.header("🎯 Promotions-Optimierung")
    st.write("""
        - 🚀 Plane und optimiere Promotionsaktionen basierend auf historischen Daten und Prognosen.
        - 💰 Maximiere Umsatz und Gewinn durch datengetriebene Entscheidungen.
        - 📅 Optimiere Budgets und Zeiträume für Aktionen.
        """)

    st.divider()

    st.write("Alle Funktionen können über die übersichtlichen Navigation am linken Seitenrand erreicht werden.")


page()
