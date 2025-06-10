import traceback

import pandas as pd
import plotly.express as px
import streamlit as st

from database.data_loader import load_data, load_multi_sales_forecast_data
from layout import with_layout
from logic.forcasting.forecast_helper import calculate_kpis
from logic.forcasting.forecaster import generate_sales_forecasts


@with_layout("📈 Verkaufsprognose-Tool (Weekly Sales)")
def page():
    st.markdown("Analysiere historische Verkaufsdaten mit KPIs, Prognosemodellen und interaktiven Diagrammen")

    df_raw, df_features, df_stores = load_data()
    if "do_prediction" not in st.session_state:
        st.session_state.do_prediction = False

    # UI Sidebar
    with st.sidebar:
        st.header("🧭 Einstellungen")
        store_ids = sorted(df_raw['StoreID'].unique())
        selected_store = st.selectbox("🏬 Store auswählen", store_ids)
        model_choices = st.multiselect("📊 Modell(e) auswählen", ["Prophet", "ARIMA", "Holt-Winters"],
                                       default=["Prophet"])
        forecast_period = st.slider("📅 Prognosezeitraum (Wochen)", 1, 52, 12)
        show_table = st.checkbox("📋 Rohdaten anzeigen", value=False)
        show_forecast_table = st.checkbox("📈 Forecast-Tabelle anzeigen", value=False)
        # Durchführen-Button
        if not st.session_state["do_prediction"]:
            st.button("Prognose starten", key="do_prediction_trigger",
                      on_click=lambda: st.session_state.update({"do_prediction": True}))

    # Daten aggregieren
    store_df = df_raw[df_raw['StoreID'] == selected_store].copy()
    store_df = store_df.groupby("Date").agg({"WeeklySales": "sum"}).reset_index()
    store_df = store_df.rename(columns={"Date": "ds", "WeeklySales": "y"})

    last_date = pd.to_datetime(store_df['ds']).max()

    # KPIs anzeigen
    total, avg, std, growth = calculate_kpis(store_df)
    st.metric("📦 Gesamtumsatz", f"{int(total):,}")
    st.metric("📊 Durchschnitt/Woche", f"{avg:.1f}")
    st.metric("📈 Volatilität", f"{std:.1f}")
    st.metric("📈 Wachstum", f"{growth:.1f}%")

    if st.session_state["do_prediction"]:
        with st.status("Führe Vorhersage durch... Bitte warten", state="running") as ui_status:
            try:
                # Führe die Vorhersage durch
                generate_sales_forecasts(store_df, forecast_period, model_choices, selected_store)
                ui_status.update(label="Vorhersage erfolgreich durchgeführt und abgespeichert!", state="complete")
            except Exception as e:
                traceback.print_exc()
                ui_status.update(label=f"Fehler: {e}", state="error")
            finally:
                # Flag zurücksetzen
                st.session_state["do_prediction"] = False

    forecasts = load_multi_sales_forecast_data(model_choices, selected_store, last_date, forecast_period)

    # Lade Prognose, falls vorhanden
    if forecasts:
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
                st.download_button(f"⬇️ Download {method}-Forecast", forecast_df.to_csv(index=False),
                                   f"{method}_forecast.csv", "text/csv")
    else:
        st.warning("⚠️ Für diese Kombination liegt keine Prognose vor.")


page()
