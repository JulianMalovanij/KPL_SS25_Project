import streamlit as st

from data_loader import load_data
from pages.promotion_optimizer.shared import create_shared_parameters, create_results, init_session, handle_optimization

st.title("Promotion Optimierung für einzelne Departments eines Stores")

# Daten laden
df_sales, df_features, df_stores = load_data()
init_session()

# Auswahl des Stores
store_ids = sorted(df_sales["StoreID"].dropna().unique())
selected_store = st.selectbox("Wähle einen Store", store_ids)

# Departments für den ausgewählten Store filtern
if selected_store:
    filtered_depts = df_sales[df_sales['StoreID'] == selected_store]['DeptID'].unique()
else:
    filtered_depts = []

# Departments auswählen (mehrere möglich)
selected_depts = st.multiselect("Wähle Departments aus", options=filtered_depts)

params = create_shared_parameters()

st.write(f"Es werden nur die Departments {", ".join(map(str, selected_depts))} des Stores {selected_store} betrachtet.")

if selected_store and selected_depts:
    # Optimierung starten – nur für ausgewählte Departments
    with st.status(f"Optimierung für {len(selected_depts)} Department(s) wird gestartet...",
                   expanded=True) as ui_status:
        df_sales_selected = df_sales[
            (df_sales['StoreID'] == selected_store) & (df_sales['DeptID'].isin(selected_depts))]

        handle_optimization(
            df_sales_selected,
            df_features,
            params,
            ui_status,
            len(selected_depts) > 1,  # parallele Optimierung nur bei mehreren Departments
            selected_stores=selected_store,
            selected_depts=selected_depts
        )
else:
    st.warning("Bitte wähle mindestens ein Department aus, um die Optimierung zu starten.")

create_results()
