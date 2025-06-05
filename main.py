import streamlit as st

st.set_page_config(
    page_title="Analyse‐Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hier werden die Seiten definiert
main_page = st.Page("pages/welcome.py", title="Home")
promo_optimizer = st.Page("pages/promo_optimizer.py", title="Promotion optimizer")
descriptive_page = st.Page("pages/descriptive_analysis.py", title="Deskriptive Analyse")

# Hier werden die Seiten den Navigationskategorien zugeordnet
# TODO: Prüfen, ob Kategorisierung wirklich notwendig ist
pg = st.navigation({
    "Home": [main_page],
    "Optimizer": [promo_optimizer],
    "KI": [],
    "Deskriptive Analyse": [descriptive_page],
})
pg.run()
