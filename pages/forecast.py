import matplotlib.pyplot as plt
import streamlit as st

from data_loader import load_data, load_forecast_data
from logic.forcasting.forecast_dept_level import run_forecast

model_option = st.sidebar.selectbox("Modell", ["Prophet", "ARIMA", "LSTM"])

st.title(f"📈 Nachfrageprognose mit {model_option}")
st.subheader("Interaktive Visualisierung pro Store und Abteilung")

df_sales, df_features, df_stores = load_data()
if "do_prediction" not in st.session_state:
    st.session_state.do_prediction = False

# Auswahl des Stores
store_ids = sorted(df_sales["StoreID"].dropna().unique())
selected_store = st.selectbox("Wähle einen Store", store_ids)

# Departments für den ausgewählten Store filtern
if selected_store:
    filtered_depts = df_sales[df_sales['StoreID'] == selected_store]['DeptID'].unique()
else:
    filtered_depts = []

# Departments auswählen
selected_dept = st.selectbox("Wähle Department aus", filtered_depts)

# Durchführen-Button
if not st.session_state["do_prediction"]:
    st.button("Prognose starten", key="do_prediction_trigger",
              on_click=lambda: st.session_state.update({"do_prediction": True}))

if model_option and selected_store and selected_dept:
    history = df_sales[(df_sales['StoreID'] == selected_store) & (df_sales['DeptID'] == selected_dept)].sort_values(
        "Date")

    if st.session_state["do_prediction"]:
        with st.status("Führe Vorhersage durch... Bitte warten", state="running") as ui_status:
            try:
                # Führe die Vorhersage durch
                run_forecast(history, model_option, selected_store, selected_dept)
                ui_status.update(label="Vorhersage erfolgreich durchgeführt und abgespeichert!", state="completed")
            except Exception as e:
                ui_status.update(label=f"Fehler: {e}", state="error")
            finally:
                # Flag zurücksetzen
                st.session_state["do_prediction"] = False

    forecast = load_forecast_data(model_option, selected_store, selected_dept)

    # Lade Prognose, falls vorhanden
    if forecast is not None and not forecast.empty:
        # Plot mit Matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["Date"], history["WeeklySales"], label="Historisch", color="black")
        ax.plot(forecast["ds"], forecast["yhat"], label="Prognose", color="blue")

        if model_option != "LSTM":
            ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                            color="blue", alpha=0.2, label="Konfidenzintervall")

        ax.set_title(f"Store {selected_store} – Dept {selected_dept}")
        ax.set_xlabel("Datum")
        ax.set_ylabel("Verkäufe")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
        st.dataframe(forecast)
    else:
        st.warning("⚠️ Für diese Kombination liegt keine Prognose vor.")
else:
    st.warning("Bitte wähle mindestens einen Algorithmus und ein Department aus, um die Prognose anzuzeigen.")
