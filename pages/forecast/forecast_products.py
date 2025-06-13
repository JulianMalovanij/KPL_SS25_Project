import traceback

import streamlit as st
from matplotlib import pyplot as plt

from database.data_loader import load_product_data, load_products_forecast_data
from layout import with_layout
from logic.forcasting.forecast_helper import (
    get_available_combinations,
    prepare_product_data,
    calculate_kpis
)
from logic.forcasting.forecaster import run_products_forecast


def _translate_identifiers(identifiers):
    # Extrahiere run_products_forecast-Parameter
    return {
        "wh_code": identifiers.get("WarehouseCode"),
        "prod_code": identifiers.get("ProductCode"),
        "cat_code": identifiers.get("ProductCategory"),
    }


def _run_and_store_forecast(df, model_choice, periods, identifiers, status_ctx_kwargs):
    """Führt Forecast durch und speichert ihn ab, mapping identifiers auf wh_code/prod_code/cat_code."""
    with st.status(**status_ctx_kwargs) as ui_status:
        try:
            run_products_forecast(df, model_choice, periods, **_translate_identifiers(identifiers))
            ui_status.update(label="✅ Prognose durchgeführt und gespeichert", state="complete")
        except Exception as e:
            traceback.print_exc()
            ui_status.update(label=f"❌ Fehler: {e}", state="error")


def _show_metrics_and_chart(df_hist, df_forecast):
    """Berechnet KPIs, zeigt Metrics und Matplotlib-Chart."""
    total, avg, std, growth = calculate_kpis(df_hist)
    st.metric("📦 Gesamt", f"{int(total):,}")
    st.metric("📊 Durchschnitt/Woche", f"{avg:.1f}")
    st.metric("📈 Volatilität", f"{std:.1f}")
    st.metric("🌱 Wachstum", f"{growth:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df_hist["ds"], df_hist["y"], label="Historisch")
    ax.plot(df_forecast["ds"], df_forecast["yhat"], label="Prognose")
    if df_forecast[['yhat_lower', 'yhat_upper']].notna().any(axis=None):
        ax.fill_between(df_forecast["ds"], df_forecast["yhat_lower"], df_forecast["yhat_upper"],
                        color="blue", alpha=0.2, label="Konfidenzintervall")
    ax.legend()
    ax.set_xmargin(0)
    st.pyplot(fig)


@with_layout("📦 Nachfrageanalyse- & Prognose-Tool")
def page():
    st.markdown("Erkunden Sie Nachfrageprognosen nach Produkt, Lager und Kategorie mit KPIs und Modellvergleich")

    # Daten laden
    df_hist, df_prod, df_cat = load_product_data()
    if "do_prod_prediction" not in st.session_state:
        st.session_state.do_prod_prediction = False

    # Auswahl-Grid vorbereiten
    (available_combinations,
     available_products,
     available_categories,
     available_cat_lager) = get_available_combinations(df_hist, df_prod, df_cat)

    model_choice = st.radio("🔍 Modell wählen", ["Prophet", "ARIMA", "Holt-Winters"], horizontal=True)

    if not st.session_state.do_prod_prediction:
        st.button("Prognose starten", key="do_prod_prediction_trigger",
                  on_click=lambda: st.session_state.update(do_prod_prediction=True))

    # Konfiguration der vier Blöcke
    blocks = [
        {
            "title": "🔢 Produkt & Lager",
            "options": available_combinations.apply(
                lambda r: f"{r['ProductCategory']} | {r['ProductCode']} | {r['WarehouseCode']}", axis=1
            ),
            "parser": lambda s: dict(
                ProductCategory=s.split("|")[0].strip(),
                ProductCode=s.split("|")[1].strip(),
                WarehouseCode=s.split("|")[2].strip()
            )
        },
        {
            "title": "🏷️ Kategorie & Lager",
            "options": available_cat_lager.apply(
                lambda r: f"{r['ProductCategory']} | {r['WarehouseCode']}", axis=1
            ),
            "parser": lambda s: dict(
                ProductCategory=s.split("|")[0].strip(),
                WarehouseCode=s.split("|")[1].strip()
            )
        },
        {
            "title": "📦 Produkt gesamt",
            "options": available_products,
            "parser": lambda s: dict(ProductCode=s)
        },
        {
            "title": "🏷️ Kategorie gesamt",
            "options": available_categories,
            "parser": lambda s: dict(ProductCategory=s)
        },
    ]

    # Layout: zwei Spalten für die ersten beiden, dann volle Breite
    col1, col2 = st.columns(2)
    for idx, block in enumerate(blocks):
        container = col1 if idx == 0 else col2 if idx == 1 else st.container()
        with container:
            st.subheader(block["title"])
            selection = st.selectbox(f"{block['title']} auswählen", block["options"], key=idx)
            if not selection:
                continue

            identifiers = block["parser"](selection)
            try:
                # historische Daten vorbereiten
                df_pre = prepare_product_data(df_hist, identifiers)

                # Forecast
                if st.session_state.do_prod_prediction:
                    _run_and_store_forecast(
                        df_pre, model_choice, 104,
                        identifiers={**identifiers},
                        status_ctx_kwargs={
                            "label": "Führe Vorhersage durch... bitte warten",
                            "state": "running"
                        }
                    )

                # Laden & Anzeigen
                df_frc = load_products_forecast_data(model_choice, **_translate_identifiers(identifiers))
                _show_metrics_and_chart(df_pre, df_frc)

            except Exception as e:
                st.error(f"Fehler: {e}")

    # Reset-Flag
    if st.session_state.do_prod_prediction:
        st.session_state.do_prod_prediction = False


page()
