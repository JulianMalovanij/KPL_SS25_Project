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
    section = st.sidebar.radio(
        "Analyse-Bereich:",
        ["Datenübersicht", "WeeklySales", "HistoricalDemand",
        "Datensatz-Vergleich", "Events"]          #  ⬅︎ neu
    )


    df_demand["OrderDemand"] = pd.to_numeric(
        df_demand["OrderDemand"], errors="coerce"
    ).fillna(0)
        # ----------------------------------------
    # 3) Datenübersicht
    # ----------------------------------------
    # ----------------------------------------
    # 3) Datenübersicht
    # ----------------------------------------
    if section == "Datenübersicht":
        st.title("1) Datenübersicht")

        # Hilfs-Funktion: Stat-Block pro DataFrame
        def overview_block(name, df, date_col=None):
            rows, cols = df.shape

            # Layout in drei Spalten
            c1, c2, c3 = st.columns(3)
            c1.metric("Zeilen", f"{rows:,}")
            c1.metric("Spalten", f"{cols}")

            # Datums­spanne, falls Spalte angegeben
            if date_col and date_col in df.columns:
                span = f"{df[date_col].min().date()} → {df[date_col].max().date()}"
            else:
                span = "—"
            c2.metric("Zeitraum", span)

            # Fehlende Werte absolut & %
            miss = df.isna().sum().sum()
            miss_pct = miss / (rows * cols) * 100
            c3.metric("Fehlende", f"{miss:,}  ({miss_pct:.2f} %)")

            # Detail-Expander
            with st.expander(f"Details zu **{name}**"):
                st.write("Datentypen:")
                st.dataframe(df.dtypes.to_frame("Typ"))
                st.write("Kopf der Tabelle:")
                st.dataframe(df.head())

        # Anzeige für jede Tabelle
        st.markdown("### Store")
        overview_block("Store", df_stores)

        st.markdown("### StoreFeature")
        overview_block("StoreFeature", df_features, date_col="Date")

        st.markdown("### WeeklySales")
        overview_block("WeeklySales", df_sales, date_col="Date")

        st.markdown("### HistoricalDemand")
        overview_block("HistoricalDemand", df_demand, date_col="Date")

        # Gesamt-Überblick numerische Statistiken
        st.markdown("---")
        st.subheader("Deskriptive Statistik (numerische Spalten)")
        tabs = st.tabs(["StoreFeature", "WeeklySales", "HistoricalDemand"])
        tabs[0].dataframe(df_features.select_dtypes("number").describe().T)
        tabs[1].dataframe(df_sales[["WeeklySales"]].describe().T)
        tabs[2].dataframe(df_demand[["OrderDemand"]].describe().T)
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

        st.markdown("---")
        st.subheader("Boxplot: WeeklySales pro StoreType")

        # StoreType anhängen
        df_ws_type = df_ws.merge(df_stores[["StoreID", "StoreType"]], on="StoreID", how="left")

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_ws_type, x="StoreType", y="WeeklySales", palette="pastel", ax=ax3)
        ax3.set_xlabel("StoreType")
        ax3.set_ylabel("WeeklySales")
        ax3.set_title("Verteilung der WeeklySales nach StoreType")
        st.pyplot(fig3)
        plt.close(fig3)

        # ------------------------------------------------------------------
        # 2.2 Ø WeeklySales pro Monat
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Durchschnittlicher WeeklySales je Monat")

        # Monat aus dem Date-Feld holen (3-stellige dt. Abkürzung)
        df_ws["Month"] = df_ws["Date"].dt.month_name(locale="de_DE").str[:3]

        # Mittelwert pro Monat berechnen und chronologisch sortieren
        month_order = ["Jan","Feb","Mär","Apr","Mai","Jun",
                    "Jul","Aug","Sep","Okt","Nov","Dez"]

        month_mean = (
            df_ws.groupby("Month")["WeeklySales"]
                .mean()
                .reindex(month_order)          # fixe Reihenfolge
                .reset_index()
        )

        fig_m, ax_m = plt.subplots(figsize=(7,3))
        sns.barplot(
            data=month_mean,
            x="Month",
            y="WeeklySales",
            palette="Greens_d",
            ax=ax_m
        )
        ax_m.set_xlabel("Monat")
        ax_m.set_ylabel("Ø WeeklySales")
        ax_m.set_title("Ø WeeklySales pro Monat")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_m)
        plt.close(fig_m)

        # ------------------------------------------------------------------
        # 2.3 Heatmap: WeeklySales pro Monat & StoreID
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Heatmap: Umsatzsaison je Store & Monat")

        df_ws["Monat"] = df_ws["Date"].dt.month_name(locale="de_DE").str[:3]
        pivot_ws = (
            df_ws.groupby(["StoreID", "Monat"])["WeeklySales"]
            .sum()
            .unstack(fill_value=0)
        )
        monat_order = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                    "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
        pivot_ws = pivot_ws.reindex(columns=monat_order)

        fig5, ax5 = plt.subplots(figsize=(9, 4))
        sns.heatmap(pivot_ws, cmap="YlGnBu", linewidths=0.5, annot=False, ax=ax5)
        ax5.set_xlabel("Monat")
        ax5.set_ylabel("StoreID")
        ax5.set_title("Summe WeeklySales pro Store & Monat")
        st.pyplot(fig5)
        plt.close(fig5)

        # ------------------------------------------------------------------
        # 2.4 Scatter: WeeklySales vs. Temperatur
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Scatter: WeeklySales vs. Temperatur")

        # Temperatur aus Features anhängen
        df_temp = df_ws.merge(
            df_features[["StoreID", "Date", "Temperature"]],
            on=["StoreID", "Date"],
            how="left"
        )
        # Stichprobe für bessere Performance
        sample_temp = df_temp.sample(n=min(4000, len(df_temp)), random_state=42)

        fig6, ax6 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=sample_temp,
            x="Temperature",
            y="WeeklySales",
            alpha=0.3,
            ax=ax6
        )
        ax6.set_xlabel("Temperatur (°F)")
        ax6.set_ylabel("WeeklySales")
        ax6.set_title("Zusammenhang: Temperatur vs. WeeklySales")
        st.pyplot(fig6)
        plt.close(fig6)
        
        # ------------------------------------------------------------------
        # 2.3 Korrelationsmatrix (WeeklySales + numerische Features)
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Korrelationsmatrix: WeeklySales & Features")

        # ❶  Merge: WeeklySales-Ausschnitt + Features-Tabelle
        df_corr = (
            df_ws.merge(
                df_features,              # enthält Temperature, FuelPrice, CPI, …
                on=["StoreID", "Date"],
                how="left"
            )
            .merge(
                df_stores[["StoreID", "StoreSize"]],  # optional: StoreSize anhängen
                on="StoreID",
                how="left"
            )
        )

        # ❷  Nur numerische Spalten auswählen
        num_cols = df_corr.select_dtypes(include="number")

        # ❸  Korrelation berechnen
        corr_mat = num_cols.corr().round(2)

        # ❹  Heatmap plotten
        fig_corr, ax_corr = plt.subplots(figsize=(9, 6))
        sns.heatmap(
            corr_mat,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.4,
            ax=ax_corr
        )
        ax_corr.set_title("Korrelationen zwischen WeeklySales & numerischen Features")
        st.pyplot(fig_corr)
        plt.close(fig_corr)

        # ------------------------------------------------------------------
        # 2.4 (Optional) Scatter-Matrix der Top-5 korrelierenden Variablen
        # ------------------------------------------------------------------
        # Top-5 absolut höchsten Korrelationen zu WeeklySales wählen
        top5 = (
            corr_mat["WeeklySales"]
            .abs()
            .sort_values(ascending=False)
            .iloc[1:6]             # erstes Element ist 1.0 (Self-Corr)
            .index
            .tolist()
        )
        pair_cols = ["WeeklySales"] + top5

        st.markdown("#### Scatter-Matrix der 5 stärksten Korrelationen")
        fig_pair = sns.pairplot(
            data=num_cols[pair_cols].sample(n=min(500, len(num_cols))),  # Sample für Geschwindigkeit
            diag_kind="kde",
            plot_kws=dict(alpha=0.3, s=20)
        )
        st.pyplot(fig_pair.fig)
        plt.close("all")

        # ------------------------------------------------------------------
        # 2.5 Zusätzliche Zusammenhänge mit Features
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Zusätzliche Analysen: CPI, Unemployment, IsHoliday, StoreSize")

        # ■ ❶ Merge mit Features und StoreSize
        df_ws_feat = (
            df_ws
            .merge(
                df_features[["StoreID","Date","CPI","Unemployment","IsHoliday"]],
                on=["StoreID","Date"], how="left"
            )
            .merge(
                df_stores[["StoreID","StoreSize"]],
                on="StoreID", how="left"
            )
        )

        # ← Neu: gemeinsame Holiday-Flag erzeugen (bool)
        df_ws_feat["HolidayFlag"] = (
            df_ws_feat.get("IsHoliday_x", False) |
            df_ws_feat.get("IsHoliday_y", False)
        )


        # ■ ❷ Barplot: Ø WeeklySales an Feiertagen vs. normalen Wochen
        fig_hol, ax_hol = plt.subplots(figsize=(6,3))
        sns.barplot(
            data=df_ws_feat,
            x="HolidayFlag",          # statt IsHoliday
            y="WeeklySales",
            ci=None,
            palette=["skyblue","salmon"],
            ax=ax_hol
        )
        ax_hol.set_xticklabels(["Normal","Feiertag"])
        ax_hol.set_xlabel("")
        ax_hol.set_ylabel("Ø WeeklySales")
        ax_hol.set_title("Durchschnittlicher Umsatz: Feiertag vs. Normal")
        st.pyplot(fig_hol)
        plt.close(fig_hol)

        # ■ ❸ Scatter: WeeklySales vs. CPI
        fig_cpi, ax_cpi = plt.subplots(figsize=(6,4))
        sns.scatterplot(
            data=df_ws_feat.sample(n=min(1000,len(df_ws_feat)), random_state=1),
            x="CPI",
            y="WeeklySales",
            alpha=0.4,
            ax=ax_cpi
        )
        ax_cpi.set_title("WeeklySales vs. CPI")
        ax_cpi.set_xlabel("Consumer Price Index (CPI)")
        ax_cpi.set_ylabel("WeeklySales")
        st.pyplot(fig_cpi)
        plt.close(fig_cpi)

        # ■ ❹ Scatter: WeeklySales vs. Unemployment
        fig_unemp, ax_unemp = plt.subplots(figsize=(6,4))
        sns.scatterplot(
            data=df_ws_feat.sample(n=min(1000,len(df_ws_feat)), random_state=2),
            x="Unemployment",
            y="WeeklySales",
            alpha=0.4,
            ax=ax_unemp
        )
        ax_unemp.set_title("WeeklySales vs. Unemployment Rate")
        ax_unemp.set_xlabel("Unemployment Rate")
        ax_unemp.set_ylabel("WeeklySales")
        st.pyplot(fig_unemp)
        plt.close(fig_unemp)

        # ■ ❺ Boxplot: WeeklySales nach StoreSize-Klassen
        st.markdown("---")
        st.subheader("WeeklySales nach StoreSize")
        fig_ss, ax_ss = plt.subplots(figsize=(6,4))
        sns.boxplot(
            data=df_ws_feat,
            x="StoreSize",
            y="WeeklySales",
            palette="vlag",
            ax=ax_ss
        )
        ax_ss.set_title("Verteilung der WeeklySales nach StoreSize")
        ax_ss.set_xlabel("StoreSize")
        ax_ss.set_ylabel("WeeklySales")
        st.pyplot(fig_ss)
        plt.close(fig_ss)



        return

    # ----------------------------------------
    # 5) HistoricalDemand Analyse
    if section == "HistoricalDemand":
        st.title("3) HistoricalDemand Analyse")
        # Filter auswählen
        cats = sorted(df_demand["ProductCategory"].unique())
        whs  = sorted(df_demand["WarehouseCode"].unique())
        sel_cat = st.sidebar.multiselect("Kategorie auswählen", cats, default=cats[:3])
        sel_wh  = st.sidebar.multiselect("Warehouse auswählen", whs, default=whs[:3])
        start2, end2 = st.sidebar.date_input(
            "Datumsspanne",
            [df_demand["Date"].min(), df_demand["Date"].max()],
            key="hd",
        )
        start2, end2 = pd.to_datetime(start2), pd.to_datetime(end2)

        df_hd = df_demand[
            df_demand["ProductCategory"].isin(sel_cat)
            & df_demand["WarehouseCode"].isin(sel_wh)
            & df_demand["Date"].between(start2, end2)
        ]
        # ← **Neu**: OrderDemand in numerisch konvertieren, sonst rollende Mittelwerte etc. brechen
        df_hd["OrderDemand"] = pd.to_numeric(df_hd["OrderDemand"], errors="coerce").fillna(0)

        st.markdown("### 3.1 Histogramm OrderDemand")
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        sns.histplot(df_hd["OrderDemand"], bins=40, kde=True, ax=ax1, color="#5A9")
        ax1.set_xlabel("Bestellmenge")
        ax1.set_ylabel("Häufigkeit")
        st.pyplot(fig1)
        plt.close(fig1)

        st.markdown("### 3.2 Monatliche Summe + 6-Monats-Ø")
        df_hd["YearMonth"] = df_hd["Date"].dt.to_period("M").dt.to_timestamp()
        monthly = df_hd.groupby("YearMonth")["OrderDemand"].sum().reset_index()
        # rolling mean funktioniert jetzt
        monthly["Rolling6"] = monthly["OrderDemand"].rolling(6).mean()

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(monthly["YearMonth"], monthly["OrderDemand"], marker="o", label="Monatssumme")
        ax2.plot(monthly["YearMonth"], monthly["Rolling6"], color="red", label="6-Monats-Ø")
        ax2.set_xlabel("Monat")
        ax2.set_ylabel("Summe OrderDemand")
        ax2.legend()
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)
        plt.close(fig2)

        # ------------------------------------------------------------------------
        # 3.3 Statische Ranglisten
        # ------------------------------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top-5 Warehouses (Gesamt-Demand)")
            top_wh = (
                df_hd.groupby("WarehouseCode")["OrderDemand"]
                .sum()
                .nlargest(5)
                .reset_index()
            )
            st.table(top_wh)

        with col2:
            st.markdown("#### Top-5 Kategorien (Gesamt-Demand)")
            top_cat = (
                df_hd.groupby("ProductCategory")["OrderDemand"]
                .sum()
                .nlargest(5)
                .reset_index()
            )
            st.table(top_cat)

        # ------------------------------------------------------------------------
        # 3.4 Heatmap Warehouse × Monat
        # ------------------------------------------------------------------------
        st.markdown("### 3.4 Heatmap Warehouse × Monat")

        df_hd["Monat"] = df_hd["Date"].dt.month_name(locale="de_DE").str[:3]
        pivot = (
            df_hd.groupby(["WarehouseCode", "Monat"])["OrderDemand"]
            .sum()
            .unstack(fill_value=0)
        )
        mon_order = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                    "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
        pivot = pivot.reindex(columns=mon_order)

        st.dataframe(pivot)  # Überblick

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0f",
            cmap="YlGnBu",
            linewidths=0.4,
            ax=ax3,
        )
        ax3.set_xlabel("Monat")
        ax3.set_ylabel("Warehouse")
        ax3.set_title("OrderDemand pro Warehouse & Monat")
        st.pyplot(fig3)
        plt.close(fig3)

        # ------------------------------------------------------------------------
        # 3.5 Top-10 Produkte nach OrderDemand (bleibt!)
        # ------------------------------------------------------------------------
        st.markdown("### 3.5 Top-10 Produkte (gefilterter Zeitraum & Auswahl)")

        top10 = (
            df_hd.groupby("ProductCode")["OrderDemand"]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        st.table(top10)

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=top10,
            x="OrderDemand",
            y="ProductCode",
            palette="pastel",
            ax=ax4,
        )
        ax4.set_title("Top-10 Produkte nach OrderDemand")
        ax4.set_xlabel("Summe OrderDemand")
        ax4.set_ylabel("ProductCode")
        st.pyplot(fig4)
        plt.close(fig4)

        # ------------------------------------------------------------------------
        # 3.6 Korrelationsmatrix (OrderDemand & kodierte Kategorien)
        # ------------------------------------------------------------------------
        st.markdown("### 3.6 Korrelationsmatrix")

        # Kategorische Spalten numerisch codieren, damit corr() funktioniert
        df_corr = df_hd.copy()
        for col in ["WarehouseCode", "ProductCategory", "ProductCode"]:
            df_corr[f"{col}_enc"] = pd.factorize(df_corr[col])[0]

        num_cols = ["OrderDemand",
                    "WarehouseCode_enc",
                    "ProductCategory_enc",
                    "ProductCode_enc"]

        corr_mat = df_corr[num_cols].corr().round(2)

        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            corr_mat,
            annot=True,
            cmap="coolwarm",
            linewidths=0.4,
            ax=ax_corr
        )
        ax_corr.set_title("Korrelationen: OrderDemand & kodierte Kategorien")
        st.pyplot(fig_corr)
        plt.close(fig_corr)


        return
    # ----------------------------------------
    # 5b) Datensatz-Vergleich (WeeklySales vs. OrderDemand)
    # ----------------------------------------
    if section == "Datensatz-Vergleich":
        st.title("3) Vergleich WeeklySales ↔ OrderDemand")

        # 1) Aggregation  ►  monatliche Summen beider Datensätze
        df_sales["YearMonth"]  = df_sales["Date"].dt.to_period("M").dt.to_timestamp()
        df_demand["YearMonth"] = df_demand["Date"].dt.to_period("M").dt.to_timestamp()

        df_demand["OrderDemand"] = pd.to_numeric(df_demand["OrderDemand"], errors="coerce").fillna(0)

        sales_month   = df_sales.groupby("YearMonth")["WeeklySales"].sum()
        demand_month  = df_demand.groupby("YearMonth")["OrderDemand"].sum()

        df_combo = pd.concat([sales_month, demand_month], axis=1).dropna()
        df_combo.columns = ["WeeklySales", "OrderDemand"]

        # 2) Linien-Plot  ►  zeitlicher Verlauf beider Größen
        st.subheader("3.1 Zeitreihe: Monatssummen beider Datensätze")

        # höchstens 120 Monatswerte anzeigen
        max_points = 120
        df_plot = df_combo.tail(max_points)

        if df_plot.empty:
            st.info("⚠️ Für den gewählten Zeitraum liegen keine Daten vor.")
        else:
            st.line_chart(df_plot)

        # 3) Scatter + Regressionslinie  ►  Zusammenhang der Monatssummen
        st.subheader("3.2 Scatter: OrderDemand vs. WeeklySales (Monat)")
        fig_c2, ax_c2 = plt.subplots(figsize=(5,4))
        sns.regplot(data=df_combo,
                    x="OrderDemand", y="WeeklySales",
                    scatter_kws=dict(alpha=0.4), ax=ax_c2)
        ax_c2.set_xlabel("OrderDemand (Summe je Monat)")
        ax_c2.set_ylabel("WeeklySales (Summe je Monat)")
        st.pyplot(fig_c2)
        plt.close(fig_c2)

        # 4) Korrelationstabelle
        st.subheader("3.3 Korrelation der Monatssummen")
        corr_val = df_combo.corr().iloc[0,1]
        st.write(f"Pearson-r = **{corr_val:.3f}**")
        st.table(df_combo.corr().round(3))

        # 5) Heatmap (monatliche Z-Scores, um Saisonmuster zu vergleichen)
        st.subheader("3.4 Saisonaler Vergleich (Z-Score-Heatmap)")
        z_df = df_combo.apply(lambda s: (s - s.mean()) / s.std())
        z_df["Monat"] = z_df.index.month_name(locale="de_DE").str[:3]
        z_df["Jahr"]  = z_df.index.year
        pivot_z = z_df.pivot_table(index="Jahr", columns="Monat",
                                values="WeeklySales")
        # gleiche Monatsreihenfolge
        mon_order = ["Jan","Feb","Mär","Apr","Mai","Jun",
                    "Jul","Aug","Sep","Okt","Nov","Dez"]
        pivot_z = pivot_z[mon_order]

        fig_c3, ax_c3 = plt.subplots(figsize=(9,3))
        sns.heatmap(pivot_z, cmap="coolwarm", center=0,
                    linewidths=.4, ax=ax_c3)
        ax_c3.set_title("Z-Score-Heatmap der WeeklySales\n"
                        "(rot = überdurchschnittlich, blau = unterdurchschnittlich)")
        st.pyplot(fig_c3)
        plt.close(fig_c3)

        return   # ⬅︎ wichtig: damit die Events-Sektion erst danach kommt



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
