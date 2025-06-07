import streamlit as st

from data_loader import load_data
from pages.promotion_optimizer.shared import create_shared_parameters, create_results, handle_optimization, init_session

st.title("Promotion Optimierung für Stores")

# Daten laden
df_sales, df_features, df_stores = load_data()
init_session()

# Auswahl mehrerer Stores
store_ids = sorted(df_sales["StoreID"].dropna().unique())
selected_stores = st.multiselect("Wähle einen oder mehrere Stores", store_ids)

params = create_shared_parameters()

st.write(f"Es werden nur die Stores {", ".join(map(str, selected_stores))} betrachtet.")

if selected_stores:
    # Optimierung starten – nur für ausgewählte Stores
    with st.status(f"Optimierung für {len(selected_stores)} Store(s) wird gestartet...", expanded=True) as ui_status:
        df_sales_selected = df_sales[df_sales.StoreID.isin(selected_stores)]

        handle_optimization(
            df_sales_selected,
            df_features,
            params,
            ui_status=ui_status,
            parallel=True,
            selected_stores=selected_stores
        )

else:
    st.warning("Bitte wähle mindestens einen Store aus, um die Optimierung zu starten.")

create_results()
