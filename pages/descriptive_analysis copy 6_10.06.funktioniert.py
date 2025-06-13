# %% pages/descriptive_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

from data_loader import load_data  # liefert df_sales, df_features, df_stores
from layout import with_layout

# Hinweis: st.set_page_config sollte in main.py stehen

@with_layout("Deskriptive Datenanalyse")
def page():
    # ----------------------------------------
    # 1) Daten laden
    # ----------------------------------------
    df_sales, df_features, df_stores = load_data()
    conn = sqlite3.connect("walmart.db")
    df_demand = pd.read_sql_query(
        "SELECT * FROM HistoricalDemand", conn, parse_dates=["Date"]
    )
    conn.close()

    # ----------------------------------------
    # 2) Sidebar: Sektionen & Einstellungen
    # ----------------------------------------
    st.sidebar.header("Einstellungen")
    section = st.sidebar.radio(
        "Analyse-Bereich:",
        ["Datenübersicht", "WeeklySales", "HistoricalDemand", "Events"]
    )

    # ----------------------------------------
    # 3) Datenübersicht
    # ----------------------------------------
    if section == "Datenübersicht":
        st.title("1) Datenübersicht")
        # Anzahl und Spalten
        st.subheader("Tabellen und Dimensionen")
        st.write(f"- Store: {df_stores.shape[0]} Zeilen, {df_stores.shape[1]} Spalten")
        st.write(f"- StoreFeature: {df_features.shape[0]} Zeilen, {df_features.shape[1]} Spalten")
        st.write(f"- WeeklySales: {df_sales.shape[0]} Zeilen, {df_sales.shape[1]} Spalten")
        st.write(f"- HistoricalDemand: {df_demand.shape[0]} Zeilen, {df_demand.shape[1]} Spalten")

        # Kopf und Datentypen
        st.subheader("Beispielinhalt & Datentypen")
        st.write("Store-Tabelle:")
        st.dataframe(df_stores.head(5))
        st.write("Datentypen StoreFeature:")
        types = df_features.dtypes.reset_index().rename(columns={'index':'Spalte',0:'Typ'})
        st.dataframe(types)

        # Fehlende Werte
        st.subheader("Fehlende Werte")
        missing = pd.DataFrame({
            'Tabelle': ['Store','StoreFeature','WeeklySales','HistoricalDemand'],
            'Fehlende': [df_stores.isna().sum().sum(),
                        df_features.isna().sum().sum(),
                        df_sales.isna().sum().sum(),
                        df_demand.isna().sum().sum()]
        })
        st.table(missing)

        # Descriptive stats numeric
        st.subheader("Deskriptive Statistik (numerisch)")
        st.write("StoreFeature:")
        st.dataframe(df_features.select_dtypes(include=['number']).describe())
        st.write("WeeklySales:")
        st.dataframe(df_sales['WeeklySales'].describe().to_frame('Value'))
        st.write("OrderDemand (HistoricalDemand):")
        st.dataframe(df_demand['OrderDemand'].describe().to_frame('Value'))
        return

    # ----------------------------------------
    # 4) WeeklySales Analyse
    # ----------------------------------------
    if section == "WeeklySales":
        st.title("2) WeeklySales Analyse")
        # Filter auswählen
        stores = sorted(df_sales['StoreID'].unique())
        depts = sorted(df_sales['DeptID'].unique())
        sel_stores = st.sidebar.multiselect('StoreID auswählen', stores, default=stores[:3])
        sel_depts  = st.sidebar.multiselect('DeptID auswählen', depts, default=depts[:2])
        dr = st.sidebar.date_input(
            'Datumsspanne WeeklySales',
            [df_sales['Date'].min(), df_sales['Date'].max()]
        )
        start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        df_ws = df_sales[
            df_sales['StoreID'].isin(sel_stores) &
            df_sales['DeptID'].isin(sel_depts) &
            df_sales['Date'].between(start, end)
        ]

        # Univariate: WeeklySales-Verteilung
        st.subheader("Univariate: Verteilung der WeeklySales")
        fig1, ax1 = plt.subplots(figsize=(6,4))
        sns.histplot(df_ws['WeeklySales'], bins=30, kde=True, ax=ax1)
        ax1.set_title('Histogramm WeeklySales')
        st.pyplot(fig1)

        # Bivariate: Zeitreihe pro Store/Dept
        st.subheader("Bivariate: Zeitreihe nach Store/Dept")
        fig2, ax2 = plt.subplots(figsize=(10,4))
        cmap = plt.cm.get_cmap('tab10')
        combos = [(s,d) for s in sel_stores for d in sel_depts]
        for idx,(s,d) in enumerate(combos):
            sub = df_ws[(df_ws['StoreID']==s)&(df_ws['DeptID']==d)]
            if sub.empty: continue
            ax2.plot(sub['Date'], sub['WeeklySales'],
                     marker='o', linestyle='-', label=f"S{s}-D{d}",
                     color=cmap(idx))
        ax2.set_xlabel('Datum')
        ax2.set_ylabel('WeeklySales')
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y'))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.legend(loc='upper left', bbox_to_anchor=(1,1))
        ax2.grid(True)
        st.pyplot(fig2)

        
        return

    # ----------------------------------------
    # 5) HistoricalDemand Analyse
    if section == "HistoricalDemand":
        st.title("3) HistoricalDemand Analyse")
        # Filter auswählen
        cats = sorted(df_demand['ProductCategory'].unique())
        whs  = sorted(df_demand['WarehouseCode'].unique())
        sel_cat = st.sidebar.multiselect('Kategorie auswählen', cats, default=cats[:2])
        sel_wh  = st.sidebar.multiselect('Warehouse auswählen', whs, default=whs[:2])
        dr2 = st.sidebar.date_input(
            'Datumsspanne HistoricalDemand',
            [df_demand['Date'].min(), df_demand['Date'].max()], key='hd')
        start2, end2 = pd.to_datetime(dr2[0]), pd.to_datetime(dr2[1])
        df_hd = df_demand[
            df_demand['ProductCategory'].isin(sel_cat) &
            df_demand['WarehouseCode'].isin(sel_wh) &
            df_demand['Date'].between(start2, end2)
        ]

        # Univariate: OrderDemand-Verteilung
        st.subheader("Univariate: Verteilung der OrderDemand")
        fig3, ax3 = plt.subplots(figsize=(6,4))
        sns.histplot(df_hd['OrderDemand'], bins=30, kde=True, ax=ax3)
        ax3.set_title('Histogramm OrderDemand')
        st.pyplot(fig3)

        # Bivariate: Heatmap Monat x Warehouse
        st.subheader("Bivariate: OrderDemand pro Monat & Warehouse")
        df_hd['Month'] = df_hd['Date'].dt.month_name().str[:3]
        pivot = df_hd.groupby(['WarehouseCode','Month'])['OrderDemand']\
                      .sum().unstack('Month')
        # Monatsreihenfolge
        mon = ['Jan','Feb','Mär','Apr','Mai','Jun','Jul','Aug','Sep','Okt','Nov','Dez']
        pivot = pivot.reindex(columns=mon)
        pivot = pivot.apply(pd.to_numeric, errors='coerce')
        fig4, ax4 = plt.subplots(figsize=(8,4))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax4)
        ax4.set_title('Summe OrderDemand pro Warehouse & Monat')
        st.pyplot(fig4)

        # Top-10 Produkte nach gesamter OrderDemand
        st.subheader("Top-10 Produkte nach gesamter OrderDemand")
        df_hd['OrderDemand'] = pd.to_numeric(df_hd['OrderDemand'], errors='coerce')
        topn = df_hd.groupby('ProductCode')['OrderDemand'].sum().nlargest(10).reset_index()
        fig5, ax5 = plt.subplots(figsize=(6,4))
        sns.barplot(data=topn, x='OrderDemand', y='ProductCode', ax=ax5)
        ax5.set_title('Top-10 Produkte nach OrderDemand')
        st.pyplot(fig5)
        return

    # ----------------------------------------
    # ----------------------------------------
    # 6) Events Analyse
    # ----------------------------------------
    if section == "Events":
        st.title("4) Events Analyse")

        # 1) Manuelle Definition klassischer Feiertage
        events = {
            "Super_Bowl":    ["2010-02-12", "2011-02-11", "2012-02-10"],
            "Labor_Day":     ["2010-09-10", "2011-09-09", "2012-09-07"],
            "Thanksgiving":  ["2010-11-26", "2011-11-25"],
            "Christmas":     ["2010-12-31", "2011-12-30"]
        }

        # 2) Flags in df_ev setzen
        df_ev = df_sales.copy()
        df_ev["IsHoliday"] = df_ev["IsHoliday"].astype(bool)
        for ev_name, ev_dates in events.items():
            df_ev[ev_name] = df_ev["Date"].isin(pd.to_datetime(ev_dates))

        # --------------------------------------------------
        # 2) Statisch: Holiday-Effekt nach StoreType
        # --------------------------------------------------
        st.subheader("Holiday-Effekt nach StoreType")
        df_ht = df_ev.merge(df_stores[["StoreID","StoreType"]],
                            on="StoreID", how="left")
        holiday_names = list(events.keys())
        means = {
            hol: df_ht[df_ht[hol]].groupby("StoreType")["WeeklySales"].mean()
            for hol in holiday_names
        }
        A_means = [means[hol].get("A",0) for hol in holiday_names]
        B_means = [means[hol].get("B",0) for hol in holiday_names]
        C_means = [means[hol].get("C",0) for hol in holiday_names]

        mean_hol = df_ht[df_ht["IsHoliday"]]["WeeklySales"].mean()
        mean_non = df_ht[~df_ht["IsHoliday"]]["WeeklySales"].mean()

        import numpy as np
        x = np.arange(len(holiday_names))
        width = 0.25

        fig1, ax1 = plt.subplots(figsize=(8,4))
        barsA = ax1.bar(x-width, A_means, width, label="Type A")
        barsB = ax1.bar(x,      B_means, width, label="Type B")
        barsC = ax1.bar(x+width, C_means, width, label="Type C")

        ax1.set_xticks(x)
        ax1.set_xticklabels(holiday_names, rotation=45)
        ax1.set_ylabel("Avg WeeklySales")
        ax1.axhline(mean_hol, color="red",   linestyle="--", label="Holiday Ø")
        ax1.axhline(mean_non, color="green", linestyle="--", label="Non-Holiday Ø")
        ax1.legend(loc="upper left", bbox_to_anchor=(1,1))

        for bars in (barsA, barsB, barsC):
            for r in bars:
                h = r.get_height()
                ax1.annotate(f"{h:,.0f}",
                             xy=(r.get_x()+r.get_width()/2, h),
                             xytext=(0,3), textcoords="offset points",
                             ha="center", va="bottom")

        st.pyplot(fig1)
        st.markdown("---")

        # --------------------------------------------------
        # 3) Interaktive Event-Analyse (Barplot & Zeitreihe)
        # --------------------------------------------------
        st.subheader("Interaktive Event-Analyse")
        sel_events = st.sidebar.multiselect(
            "Welche Events anzeigen?",
            options=holiday_names,
            default=holiday_names
        )

        # 3a) Barplot: Ø WeeklySales pro ausgewähltem Event
        overall = (
            df_ev
            .melt(id_vars=["WeeklySales"],
                  value_vars=sel_events,
                  var_name="Event", value_name="IsEvent")
            .query("IsEvent == True")
            .groupby("Event")["WeeklySales"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig2, ax2 = plt.subplots(figsize=(6,3))
        sns.barplot(data=overall, x="WeeklySales", y="Event", ax=ax2, palette="pastel")
        ax2.set_xlabel("Ø WeeklySales")
        ax2.set_title("Average WeeklySales pro Event")
        st.pyplot(fig2)

        st.markdown("---")

        # 3b) Zeitreihe: WeeklySales über Zeit für ausgewählte Events & Non-Event
        fig3, ax3 = plt.subplots(figsize=(10,4))
        for ev in sel_events:
            ser = (df_ev[df_ev[ev]]
                   .groupby("Date")["WeeklySales"]
                   .mean()
                   .rename(ev))
            ax3.plot(ser.index, ser.values, marker='o', label=ev)

        non = (df_ev[~df_ev[sel_events].any(axis=1)]
               .groupby("Date")["WeeklySales"]
               .mean()
               .rename("Non_Event"))
        ax3.plot(non.index, non.values, color="gray", linestyle="--", label="Non_Event")

        ax3.set_xlabel("Datum")
        ax3.set_ylabel("Ø WeeklySales")
        ax3.legend(loc="upper left", bbox_to_anchor=(1,1))
        ax3.grid(True)
        st.pyplot(fig3)

        return
page()




# %%

# %%
