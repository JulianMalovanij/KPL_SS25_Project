import streamlit as st


def render_footer():
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: grey; font-size: 0.9em;'>"
        "<p>Verkaufsanalyse-App | Projekt für das Modul KPL an der OTH Regensburg im SoSe 2025 | Gruppe 10</p>"
        "<p>Nikolai Mallet, Julian Malovanij, Bastian Rapps, Sebastian Schall | Erstellt mit ❤️ und Streamlit</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def with_layout(title=None, show_footer=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if title:
                st.title(title)
            func(*args, **kwargs)
            if show_footer:
                render_footer()

        return wrapper

    return decorator
