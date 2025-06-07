import traceback

import streamlit as st

import import_product_db

if "tool_import_db" not in st.session_state:
    st.session_state["tool_import_db"] = False

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
