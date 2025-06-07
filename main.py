import streamlit as st

# ↑ Ganz oben die App‐Konfiguration (NICHT in den Unterseiten)
st.set_page_config(
    page_title="Analyse‐Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Anschließend eure Navigation:
main_page = st.Page("pages/welcome.py", title="Home")
promo_optimizer = st.Page("pages/promo_optimizer.py", title="Promotion optimizer")
descriptive_page = st.Page("pages/descriptive_analysis.py", title="Deskriptive Analyse")
forecast_page = st.Page("pages/forecast.py", title="Vorhersage")

pg = st.navigation({
    "Home": [main_page],
    "Optimizer": [promo_optimizer],
    "KI": [forecast_page],
    "Deskriptive Analyse": [descriptive_page],
})
pg.run()
