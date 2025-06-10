import numpy as np
import pandas as pd
import streamlit as st

from database.data_loader import load_full_sales_forecast_data
from logic.optimization.optimizations import run_promotion_sales_optimization_all
from logic.optimization.visualizations import prepare_solution_data, plot_sales_boost

# Manuelles Mapping von Parameternamen zu leserlichen Beschriftungen
PARAMETER_LABELS = {
    "promo_cost": "Kosten (% vom Umsatz)",
    "promo_boost": "Maximaler Boost (%)",
    "promo_scaling": "Wirkungserholung (%)",
    "promo_decay": "Wirkungsnachlass (%)",
    "solver_timeout": "Solver-Timeout (Sekunden)",
    "use_prediction": "Verwende Vorhersagen, falls vorhanden",
    "selected_model": "Ausgewähltes Vorhersagenmodell"
}


def init_session():
    if "promo_state" not in st.session_state:
        st.session_state["promo_state"] = create_promo_state()


def create_promo_state(df_solution=None, status=None, run_opts=False, selected_stores=None, selected_depts=None,
                       params=None):
    return {
        "df_solution": df_solution,
        "status": status,
        "run_optimization": run_opts,
        "selected_stores": selected_stores,
        "selected_depts": selected_depts,
        "params": params,
    }


def create_shared_parameters():
    # Sidebar: Parameter
    st.sidebar.header("Parameter")

    # Weitere Parameter
    promo_cost = st.sidebar.number_input("Kosten pro Promotion in % vom Umsatz", value=5.0, step=0.1, min_value=0.0,
                                         max_value=100.0)
    promo_boost = st.sidebar.number_input("Maximaler Boost je Promotion (performanceabhängig) in %", value=15.0,
                                          step=0.1,
                                          min_value=0.0, max_value=100.0)
    promo_scaling = st.sidebar.number_input("Wirkungserholung je Woche ohne Promotion in %", value=25.0, step=0.1,
                                            min_value=0.0, max_value=100.0)
    promo_decay = st.sidebar.number_input("Wirkungsnachlass je Woche mit wiederholter Promotion in %", value=40.0,
                                          step=0.1,
                                          min_value=0.0, max_value=100.0)

    # Solver-Einstellungen
    st.sidebar.divider()
    st.sidebar.header("Solver Einstellungen")
    use_prediction = st.sidebar.checkbox("Verwende Vorhersage anstelle historischer Daten, falls verfügbar", value=True)
    selected_model = None
    if use_prediction:
        st.sidebar.info(
            "Es werden die gespeicherten Vorhersagen aus dem ausgewählten Modell verwendet. Sie können auf der Vorhersage-Seite generiert werden.")
        selected_model = st.sidebar.selectbox("Vorhersagenmodell", ["Prophet", "ARIMA", "Holt-Winters"])
    solver_timeout = st.sidebar.number_input("Solver-Timeout in Sekunden (kann Güte reduzieren)", value=150, step=1,
                                             min_value=0)

    # Optimierungs-Button
    if not st.session_state["promo_state"]["run_optimization"]:
        st.button("Optimierung starten", key="run_optimization_trigger",
                  on_click=lambda: st.session_state["promo_state"].update({"run_optimization": True}))

    # Hauptbeschreibung
    st.write("### Finde eine optimale Verteilung für die Promotionen...")
    st.write(f"Jede Promotion kostet {promo_cost}% des Umsatzes der Woche.")
    st.write(
        f"Jede Promotion erhöht die Einnahmen um maximal {promo_boost}%, wobei umsatzschwache Wochen bessere Werte als umsatzstarke Wochen erzielen.")
    st.write(f"Jede wiederholte Promotion verliert {promo_decay}% ihrer Wirkung.")
    st.write(f"Jede Woche ohne Promotion lässt die Wirkung um {promo_scaling}% wieder ansteigen.")

    return create_params_state(promo_cost, promo_boost, promo_scaling, promo_decay, solver_timeout, use_prediction,
                               selected_model)


def create_params_state(promo_cost=None, promo_boost=None, promo_scaling=None, promo_decay=None, solver_timeout=None,
                        use_prediction=None, selected_model=None):
    return {
        "promo_cost": promo_cost,
        "promo_boost": promo_boost,
        "promo_scaling": promo_scaling,
        "promo_decay": promo_decay,
        "solver_timeout": solver_timeout,
        "use_prediction": use_prediction,
        "selected_model": selected_model,
    }


def handle_optimization(df_sales, df_features, params, ui_status, parallel, selected_stores=None, selected_depts=None):
    state = st.session_state["promo_state"]

    # Nur ausführen, wenn angefordert
    if not state["run_optimization"]:
        if ui_status:
            ui_status.update(label="Gespeicherte Daten werden angezeigt...")
        return

    # Filter Stores und Depts
    df_sales = filter_sales(df_sales, selected_stores, selected_depts)

    # Verwende Vorhersagedaten, falls vorhanden
    if params["use_prediction"]:
        df_pred = load_full_sales_forecast_data(params["selected_model"])
        df_pred = filter_sales(df_pred, selected_stores, selected_depts)
        df_sales = merge_forecast_with_sales(df_sales, df_pred)

    df_solution, status = run_optimization(
        df_sales,
        df_features,
        params,
        ui_status,
        parallel
    )

    # Ergebnisse speichern
    state["df_solution"] = df_solution
    state["status"] = status
    state["run_optimization"] = False
    state["selected_stores"] = selected_stores
    state["selected_depts"] = selected_depts
    state["params"] = params


def run_optimization(df_sales, df_features, params, ui_status, parallel):
    return run_promotion_sales_optimization_all(
        df_sales,
        df_features,
        cost_rate=params["promo_cost"] * 0.01,
        boost_max=params["promo_boost"] * 0.01,
        recovery_rate=params["promo_scaling"] * 0.01,
        decay_factor=1 - params["promo_decay"] * 0.01,
        ui_status=ui_status,
        parallel=parallel,
        solver_timeout=params["solver_timeout"]
    )


def create_results(df_solution=None, status=None, selected_stores=None, selected_depts=None, params=None):
    st.divider()

    # Lade die Daten aus dem state, falls vorhanden
    state = st.session_state["promo_state"]
    if state is not None:
        if df_solution is None:
            df_solution = state["df_solution"]
        if status is None:
            status = state["status"]
        if selected_stores is None:
            selected_stores = state["selected_stores"]
        if selected_depts is None:
            selected_depts = state["selected_depts"]
        if params is None:
            params = state["params"]

    # Lösung ist nicht definiert → Abbruch
    if df_solution is None or status is None:
        st.write("Keine gespeicherten Optimierungsdaten. Stelle die Parameter ein und führe eine Optimierung durch.")
        return

    # Ergebnis anzeigen
    st.write("### Ergebnisübersicht")

    with st.expander("Angewandte Optimierungsparameter anzeigen"):
        if selected_stores is not None:
            st.write(f"Die folgenden Ergebnisse gelten für Store(s): {format_selection(selected_stores)}")
        if selected_depts is not None:
            st.write(
                f"Von den Stores wurden die folgenden Department(s) betrachtet: {format_selection(selected_depts)}")
        if params is not None:
            # Parameterdaten aufbereiten
            param_rows = []
            for key, value in params.items():
                readable_name = PARAMETER_LABELS.get(key, key.replace("_", " ").capitalize())
                param_rows.append({"Parameter": readable_name, "Wert": value})

            df_params = pd.DataFrame(param_rows)
            df_params["Wert"] = df_params["Wert"].astype(str)
            st.write(f"Die Ausführung wurde durch die folgenden Parameter definiert: ")
            st.dataframe(df_params)

    # Status-Tabelle für Übersicht generieren
    status_df = pd.DataFrame([
        {
            "StoreID": df['StoreID'].iloc[0],
            "DeptID": df['DeptID'].iloc[0],
            "Status": stat
        }
        for df, stat in status
    ])

    # Status-Tabelle anzeigen
    st.dataframe(status_df, use_container_width=True)

    # Gesamtlösungsdaten vorbereiten
    df_prepared = prepare_solution_data(df_solution)
    if df_prepared.empty:
        st.warning("Keine Promotions in der Lösung.")
        return

    # Alle Stores mit Lösungen
    stores_all = sorted(df_prepared['StoreID'].unique())
    if not stores_all:
        st.warning("Keine gültigen Store-Daten vorhanden.")
        return

    st.write("### Visualisierung nach Store")
    store_tabs = st.tabs([f"Store {store}" for store in stores_all])

    for i, store in enumerate(stores_all):
        with store_tabs[i]:
            df_store = df_prepared[df_prepared['StoreID'] == store]
            departments_all = sorted(df_store['DeptID'].unique())

            selected_departments = st.multiselect(
                f"Departments für Store {store} auswählen:",
                departments_all,
                default=departments_all[:3],
                key=f"dept_select_store_{store}"
            )

            df_filtered = df_store[df_store['DeptID'].isin(selected_departments)]

            for dept in sorted(df_filtered['DeptID'].unique()):
                df_dept = df_filtered[df_filtered['DeptID'] == dept]
                st.subheader(f"Department {dept}")
                fig = plot_sales_boost(df_dept, store, dept)
                st.pyplot(fig)
                st.dataframe(df_dept, use_container_width=True)


def format_selection(selection):
    if selection is None:
        return ""
    if isinstance(selection, np.ndarray):
        selection = selection.tolist()
    if isinstance(selection, list):
        return ", ".join(map(str, selection))
    return str(selection)


def merge_forecast_with_sales(df_sales, df_pred):
    # Erstelle einen zusammengesetzten Schlüssel aus Date, StoreID und DeptID
    sales_keys = set(zip(df_sales["Date"], df_sales["StoreID"], df_sales["DeptID"]))
    pred_keys = list(zip(df_pred["ds"], df_pred["StoreID"], df_pred["DeptID"]))

    # Nur zukünftige Vorhersagen behalten (nicht in df_sales enthalten)
    future_preds = df_pred.loc[[key not in sales_keys for key in pred_keys]].copy()
    future_preds.rename(columns={"ds": "Date", "yhat": "WeeklySales"}, inplace=True)

    future_preds["IsHoliday"] = False

    # DataFrames kombinieren
    df_combined = pd.concat([df_sales, future_preds[df_sales.columns]], ignore_index=True)
    df_combined.sort_values(["StoreID", "DeptID", "Date"], inplace=True)

    return df_combined


def filter_sales(df_sales, selected_stores=None, selected_depts=None):
    # Filter für StoreID
    if selected_stores is not None:
        if not isinstance(selected_stores, (list, tuple, set)):
            selected_stores = [selected_stores]
        df_sales = df_sales[df_sales["StoreID"].isin(selected_stores)]

    # Filter für DeptID
    if selected_depts is not None:
        if not isinstance(selected_depts, (list, tuple, set)):
            selected_depts = [selected_depts]
        df_sales = df_sales[df_sales["DeptID"].isin(selected_depts)]

    return df_sales
