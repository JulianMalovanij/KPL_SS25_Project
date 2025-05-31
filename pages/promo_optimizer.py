import streamlit as st

from data_loader import load_data
from optimizations import run_promotion_sales_optimization
from visualizations import plot_promotion_optimization

st.set_page_config(page_title="Promotion Optimierung", layout="wide")
st.title("Promotion Optimierung mit Walmart-Daten")
df_sales, df_features, df_stores = load_data()

st.sidebar.header("Parameter")
budget = st.sidebar.number_input("Budget (Gesamt)", value=50000, step=1000, min_value=0)
max_promos_per_week = st.sidebar.number_input("Max. Promotions pro Woche", value=50, step=1, min_value=1)
promo_cost = st.sidebar.number_input("Kosten pro Promotion", value=1000, step=100, min_value=0)
min_store_size = st.sidebar.number_input("Min. Ladenfläche für Promotionen", value=100_000, step=1000, min_value=0)
promo_boost = st.sidebar.number_input("Boost je Promotion in %", value=10.0, step=0.1, min_value=0.0,
                                      max_value=100.0)

st.write("### Finde eine optimale Verteilung für die Promotionen...")
st.write(f"{max_promos_per_week} Promotionen pro Woche für je {promo_cost}.")
st.write(f"Maximales Budget von {budget} für Läden mit einer Fläche von mindestens {min_store_size}.")
st.write(f"Jede Promotion erhöht die Einnahmen um {promo_boost}%.")

with st.status("Optimierung wird gestartet...", expanded=True) as ui_status:
    df_solution, status = run_promotion_sales_optimization(df_sales, df_features, df_stores,
                                                           min_store_size=min_store_size,
                                                           budget=budget,
                                                           max_promos_per_week=max_promos_per_week,
                                                           promo_cost=promo_cost,
                                                           promo_boost=promo_boost * 0.01 + 1,
                                                           ui_status=ui_status)
st.divider()
st.write("### Ergebnis")
st.write(f"**Optimierungsstatus:** {status}")
st.dataframe(df_solution)
st.pyplot(plot_promotion_optimization(df_solution))
