import streamlit as st

st.set_page_config(
    page_title="Analyse‐Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hier werden die Seiten definiert
main_page = st.Page("pages/welcome.py", title="Home")
promo_optimizer_stores = st.Page("pages/promotion_optimizer/promo_optimizer_stores.py",
                                 title="Store Promotion optimizer")
promo_optimizer_depts = st.Page("pages/promotion_optimizer/promo_optimizer_departments.py",
                                title="Department Promotion optimizer")
util_page = st.Page("pages/util.py", title="Verwaltung")
descriptive_page = st.Page("pages/descriptive_analysis.py", title="Deskriptive Analyse")
forecast_store_page = st.Page("pages/forecast/forecast_stores.py", title="Verkaufsprognose-Tool (Weekly Sales)")
forecast_dept_page = st.Page("pages/forecast/forecast_departments.py", title="Verkaufsprognose (Departments)")
forecast_products_page = st.Page("pages/forecast/forecast_products.py", title="Nachfrageanalyse- & Prognose-Tool")

# Hier werden die Seiten den Navigationskategorien zugeordnet
pg = st.navigation({
    "Home": [main_page, util_page],
    "Optimizer": [promo_optimizer_stores, promo_optimizer_depts],
    "KI": [forecast_store_page, forecast_dept_page, forecast_products_page],
    "Deskriptive Analyse": [descriptive_page],
})
pg.run()
