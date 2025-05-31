import streamlit as st

main_page = st.Page("pages/welcome.py", title="Home")
promo_optimizer = st.Page("pages/promo_optimizer.py", title="Promotion optimizer")

pg = st.navigation({"Home": [main_page], "Optimizer": [promo_optimizer], "KI": [], "Deskriptive Analyse": []})
pg.run()
