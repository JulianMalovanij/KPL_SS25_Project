import traceback

import streamlit as st

import import_product_db
from data_loader import load_data
from logic.forcasting.forecast_dept_level import run_forecast

available_models = ["Prophet", "ARIMA", "LSTM"]

if "tool_import_db" not in st.session_state:
    st.session_state["tool_import_db"] = False
if "tool_predict" not in st.session_state:
    st.session_state["tool_predict"] = False


def do_prediction(models):
    df_sales, df_features, df_stores = load_data()

    # Alle eindeutigen Kombinationen aus StoreID und DeptID
    combinations = df_sales[["StoreID", "DeptID"]].drop_duplicates()

    static_str = "\n Dies wird extrem lange dauern... Bitte warten!"
    for i, model_option in enumerate(models):
        model_str = f"(Model {i + 1}/{len(models)}: {model_option})"
        with st.status(f"Führe **alle** Vorhersagen aus {model_str}: Startup... {static_str}",
                       state="running") as ui_status:
            try:
                for _, row in combinations.iterrows():
                    store_id = row["StoreID"]
                    dept_id = row["DeptID"]

                    pred_str = f"Store {store_id} / Dept {dept_id}"

                    # Historische Zeitreihe für diesen Store und dieses Department
                    history = df_sales[
                        (df_sales["StoreID"] == store_id) &
                        (df_sales["DeptID"] == dept_id)
                        ][["Date", "WeeklySales"]].copy()

                    # Überspringe leere oder zu kurze Zeitreihen
                    if len(history) < 10:
                        continue

                    ui_status.update(label=f"Führe **alle** Vorhersagen aus {model_str}: {pred_str}... {static_str}")

                    run_forecast(history, model_option, store_id, dept_id)

                ui_status.update(label=f"Alle Vorhersagen {model_str} abgeschlossen und gespeichert.", state="complete")

            except Exception as e:
                traceback.print_exc()
                ui_status.update(label=f"Fehler: {e}", state="error")

    st.session_state["tool_predict"] = False


# ---------- Seite ----------
st.title("Verwaltungstools")

st.warning("Diese Tools können das Programm empfindlich stören, unbrauchbar machen oder extrem lange andauern.")

# Logik
# Importiere die Datenbank
if st.session_state["tool_import_db"]:
    with st.status("Lösche alte Datenbank... Bitte warten!", state="running") as ui_status:
        try:
            import_product_db.drop()
            ui_status.update(label="Importiere die Datensätze... Bitte warten!")
            import_product_db.do_import()
            ui_status.update(label="Import abgeschlossen.", state="complete")

        except Exception as e:
            traceback.print_exc()
            ui_status.update(label=f"Fehler: {e}", state="error")

        finally:
            st.session_state["tool_import_db"] = False

st.write("### Datenbank aufräumen")
# Datenbank-Button
if not st.session_state["tool_import_db"]:
    st.button("Datenbank importieren", key="do_import_db_trigger",
              on_click=lambda: st.session_state.update({"tool_import_db": True}))

st.divider()
st.write("### Prognosen neu erstellen")
st.write("** Wird je nach Modell extrem lange dauern **")

# Modelle auswählen (mehrere möglich)
selected_models = st.multiselect("Wähle Departments aus", options=available_models, default=available_models[:3])

# Prognose-Button
if not st.session_state["tool_predict"]:
    st.button("Prognose starten", key="do_tool_prediction_trigger",
              on_click=lambda: st.session_state.update({"tool_predict": True}))

# Führe alle Vorhersagen aus
if st.session_state["tool_predict"]:
    do_prediction(selected_models)
