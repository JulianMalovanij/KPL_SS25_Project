# %%

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from data_loader import load_data
from layout import with_layout


@with_layout("Deskriptive Datenanalyse")
def page():
    # ----------------------------------------
    # 1) Seite konfigurieren
    # ----------------------------------------
    # → Titel über Annotation gesetzt

    # ----------------------------------------
    # 2) Daten laden (rein lesend über cache)
    # ----------------------------------------
    df_sales, df_features, df_stores = load_data()

    # ----------------------------------------
    # 3) Erste Grundstatistiken zeigen
    # ----------------------------------------
    st.subheader("1. Übersicht der Tabellen")
    st.write("- **Store** (Anzahl Stores: {})".format(len(df_stores)))
    st.write("- **StoreFeature** (Anzahl Zeilen: {})".format(len(df_features)))
    st.write("- **WeeklySales** (Anzahl Zeilen: {})".format(len(df_sales)))

    st.markdown("---")
    st.subheader("2. Beispiel: Inhalt der Store-Tabelle")
    st.dataframe(df_stores.head(10))  # zeigt die ersten 10 Zeilen von df_stores

    st.markdown("---")
    st.subheader("3. Deskriptive Statistiken der StoreFeature-Tabelle")
    st.write(df_features.select_dtypes(include=["float64", "int64"]).describe())

    # ----------------------------------------
    # 4) Univariate Exploration (Nur eine Variable)
    # ----------------------------------------

    ## 4.1 Zentrale Tendenzen & Streuungsmaße für WeeklySales
    st.markdown("---")
    st.subheader("4.1 Zentrale Tendenzen & Streuungsmaße für WeeklySales")
    weekly = df_sales["WeeklySales"]
    mean_val = weekly.mean()
    median_val = weekly.median()
    std_val = weekly.std()
    q1_val = weekly.quantile(0.25)
    q3_val = weekly.quantile(0.75)

    st.write(f"- **Mittelwert (Mean):** {mean_val:,.2f}")
    st.write(f"- **Median:** {median_val:,.2f}")
    st.write(f"- **Standardabweichung (Std):** {std_val:,.2f}")
    st.write(f"- **1. Quartil (25%):** {q1_val:,.2f}")
    st.write(f"- **3. Quartil (75%):** {q3_val:,.2f}")

    ## 4.2 Histogramm mit Dichtekurve (KDE)
    st.markdown("---")
    st.subheader("4.2 Histogramm mit Dichtekurve (KDE)")
    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data=df_sales,
        x="WeeklySales",
        bins=40,
        kde=True,
        color="skyblue",
        ax=ax_hist
    )
    ax_hist.set_title("Verteilung der WeeklySales")
    ax_hist.set_xlabel("WeeklySales")
    ax_hist.set_ylabel("Anzahl Beobachtungen")
    ax_hist.set_xscale("log")
    ax_hist.set_xmargin(0)
    st.pyplot(fig_hist)

    st.markdown("---")
    st.subheader("4.3 Verteilung der Store-Typen")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df_stores, x="StoreType", ax=ax)
    ax.set_title("Anzahl der Stores je StoreType")
    ax.set_xlabel("StoreType")
    ax.set_ylabel("Anzahl")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("4.4 Zeitreihe: Gesamtumsatz über die Zeit")
    # ► Wir fassen pro Datum alle Stores zusammen
    umsatz_pro_datum = df_sales.groupby("Date")["WeeklySales"].sum().reset_index()

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=umsatz_pro_datum, x="Date", y="WeeklySales", ax=ax3)
    ax3.set_title("Gesamter wöchentlicher Umsatz über die Zeit")
    ax3.set_xlabel("Datum")
    ax3.set_ylabel("WeeklySales")
    ax3.set_xmargin(0)
    st.pyplot(fig3)

    # ----------------------------------------
    # 5) Bivariate Exploration (Zwei Variablen)
    # ----------------------------------------

    ## 5.1 Boxplot: WeeklySales je StoreType
    st.markdown("---")
    st.subheader("5.1 Boxplot: WeeklySales je StoreType")
    df_merged_type = df_sales.merge(df_stores, on="StoreID", how="left")
    fig_box, ax_box = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df_merged_type,
        x="StoreType",
        y="WeeklySales",
        palette="pastel",
        ax=ax_box
    )
    ax_box.set_title("Distribution von WeeklySales nach StoreType")
    ax_box.set_xlabel("StoreType")
    ax_box.set_ylabel("WeeklySales")
    ax_box.set_yscale("log")
    st.pyplot(fig_box)

    ## 5.2 Scatterplot: Temperature vs. WeeklySales
    st.markdown("---")
    st.subheader("5.2 Scatterplot: Temperature vs. WeeklySales")
    df_temp_sales = df_sales.merge(
        df_features[["StoreID", "Date", "Temperature"]],
        on=["StoreID", "Date"],
        how="left"
    )
    # Stichprobe für Übersichtlichkeit
    df_sample = df_temp_sales.sample(2000, random_state=42)
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df_sample,
        x="Temperature",
        y="WeeklySales",
        alpha=0.3,
        ax=ax_scatter
    )
    ax_scatter.set_title("Zusammenhang: Temperature vs. WeeklySales")
    ax_scatter.set_xlabel("Temperature (°F)")
    ax_scatter.set_ylabel("WeeklySales")
    ax_scatter.set_yscale("log")
    st.pyplot(fig_scatter)

    ## 5.3 Barplot: Durchschnittlicher WeeklySales je StoreType
    st.markdown("---")
    st.subheader("5.3 Barplot: Durchschnittlicher WeeklySales je StoreType")
    avg_sales = df_sales.groupby("StoreID")["WeeklySales"].mean().reset_index()
    avg_sales = avg_sales.merge(df_stores[["StoreID", "StoreType"]], on="StoreID", how="left")
    avg_per_type = avg_sales.groupby("StoreType")["WeeklySales"].mean().reset_index()

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=avg_per_type,
        x="StoreType",
        y="WeeklySales",
        palette="muted",
        ax=ax_bar
    )
    ax_bar.set_title("Mittlerer WeeklySales pro StoreType")
    ax_bar.set_xlabel("StoreType")
    ax_bar.set_ylabel("Durchschnittlicher WeeklySales")
    st.pyplot(fig_bar)

    # ----------------------------------------
    # 6) Korrelationen (Mehrere Variablen)
    # ----------------------------------------

    st.markdown("---")
    st.subheader("6. Korrelationsmatrix numerischer Variablen")
    df_corr_base = df_sales.merge(df_features, on=["StoreID", "Date"], how="left")
    numeric_cols = [
        "WeeklySales", "Temperature", "FuelPrice", "CPI", "Unemployment",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"
    ]
    # Nur numerische Spalten auswählen
    df_corr = df_corr_base[numeric_cols].copy()
    corr_matrix = df_corr.corr()

    fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax_heat
    )
    ax_heat.set_title("Korrelationsmatrix numerischer Variablen")
    st.pyplot(fig_heat)

    # ----------------------------------------
    # 7) Zeitreihen-Analysen & Saisonalität
    # ----------------------------------------

    ## 7.1 Jahresvergleich: Gesamtumsatz pro Jahr
    st.markdown("---")
    st.subheader("7.1 Jahresvergleich: Gesamtumsatz pro Jahr")
    df_sales["Year"] = df_sales["Date"].dt.year
    sales_per_year = df_sales.groupby("Year")["WeeklySales"].sum().reset_index()

    fig_year, ax_year = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=sales_per_year,
        x="Year",
        y="WeeklySales",
        palette="Blues_d",
        ax=ax_year
    )
    ax_year.set_title("Gesamter WeeklySales pro Jahr")
    ax_year.set_xlabel("Jahr")
    ax_year.set_ylabel("Gesamtumsatz")
    st.pyplot(fig_year)

    ## 7.2 Monats-Heatmap: Umsatz je StoreType
    st.markdown("---")
    st.subheader("7.2 Monats-Heatmap: Umsatz je StoreType")
    df_temp2 = df_sales.copy()
    df_temp2["Month"] = df_temp2["Date"].dt.month
    df_temp2 = df_temp2.merge(df_stores[["StoreID", "StoreType"]], on="StoreID", how="left")
    pivot_month = df_temp2.groupby(["StoreType", "Month"])["WeeklySales"].sum().unstack()

    fig_month, ax_month = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot_month,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax_month
    )
    ax_month.set_title("Umsatz (Summe) pro Monat & StoreType")
    ax_month.set_xlabel("Monat")
    ax_month.set_ylabel("StoreType")
    st.pyplot(fig_month)

    # ----------------------------------------
    # 8) Feiertagseffekte
    # ----------------------------------------

    ## 8.1 Durchschnittlicher Umsatz: Feiertag vs. Normal
    st.markdown("---")
    st.subheader("8.1 Durchschnittlicher Umsatz: Feiertag vs. Normal")

    # A) Ursprüngliches IsHoliday aus df_sales entfernen
    df_sales_ohne_h = df_sales.drop(columns=["IsHoliday"])

    # B) Merge mit IsHoliday aus df_features (so entsteht genau eine Spalte "IsHoliday")
    df_holiday = df_sales_ohne_h.merge(
        df_features[["StoreID", "Date", "IsHoliday"]],
        on=["StoreID", "Date"],
        how="left"
    )

    # C) Gruppieren nach IsHoliday und Durchschnitt berechnen
    avg_holiday = df_holiday.groupby("IsHoliday")["WeeklySales"].mean().reset_index()

    # D) Labels umbenennen (0 = Nicht-Feiertag, 1 = Feiertag)
    avg_holiday["Typ"] = avg_holiday["IsHoliday"].map({0: "Nicht-Feiertag", 1: "Feiertag"})

    # E) Barplot
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=avg_holiday,
        x="Typ",
        y="WeeklySales",
        palette=["salmon", "limegreen"],
        ax=ax7
    )
    ax7.set_title("Durchschnittlicher WeeklySales: Feiertag vs. Normal")
    ax7.set_xlabel("")
    ax7.set_ylabel("Durchschnittlicher Umsatz")
    st.pyplot(fig7)

    ## 8.2 Anteil Holiday-Umsatz am Jahresgesamterlös (%)
    st.markdown("---")
    st.subheader("8.2 Anteil Holiday-Umsatz am Jahresgesamterlös (%)")

    # C) Jahr aus Datum extrahieren
    df_holiday["Year"] = df_holiday["Date"].dt.year

    # D) Umsatz nach Jahr und IsHoliday summieren
    sales_by_year_hol = (
        df_holiday
        .groupby(["Year", "IsHoliday"])["WeeklySales"]
        .sum()
        .reset_index()
    )

    # E) Pivotieren: Spalten "NormalSales" und "HolidaySales"
    pivot_hol = sales_by_year_hol.pivot(
        index="Year",
        columns="IsHoliday",
        values="WeeklySales"
    ).fillna(0)
    pivot_hol.columns = ["NormalSales", "HolidaySales"]

    # F) Anteil (%) berechnen
    pivot_hol["HolidayRatio"] = (pivot_hol["HolidaySales"] / (
            pivot_hol["NormalSales"] + pivot_hol["HolidaySales"])) * 100

    # G) Liniendiagramm
    fig8, ax8 = plt.subplots(figsize=(6, 4))
    sns.lineplot(
        data=pivot_hol.reset_index(),
        x="Year",
        y="HolidayRatio",
        marker="o",
        ax=ax8
    )
    ax8.set_title("Anteil Holiday-Umsatz (%) nach Jahr")
    ax8.set_xlabel("Jahr")
    ax8.set_ylabel("Anteil in %")
    ax8.set_ylim(0, pivot_hol["HolidayRatio"].max() * 1.1)
    ax8.set_xmargin(0)
    ax8.set_xticklabels(plt.xticks()[0].astype(int))
    st.pyplot(fig8)

    # ----------------------------------------
    # 9) Standort-Analyse
    # ----------------------------------------

    ## 9.1 Top-10 Stores nach Gesamtumsatz
    st.markdown("---")
    st.subheader("9.1 Top-10 Stores nach Gesamtumsatz")
    total_per_store = df_sales.groupby("StoreID")["WeeklySales"].sum().reset_index()
    top10 = total_per_store.sort_values("WeeklySales", ascending=False).head(10)

    fig9, ax9 = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=top10,
        x="StoreID",
        y="WeeklySales",
        palette="viridis",
        ax=ax9
    )
    ax9.set_title("Top-10 Stores nach Gesamtumsatz")
    ax9.set_xlabel("StoreID")
    ax9.set_ylabel("Gesamtumsatz")
    st.pyplot(fig9)

    ## 9.2 WeeklySales nach StoreSize (Boxplot)
    st.markdown("---")
    st.subheader("9.2 WeeklySales nach StoreSize (Boxplot)")
    df_size = df_sales.merge(df_stores[["StoreID", "StoreSize"]], on="StoreID", how="left")

    fig10, ax10 = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=df_size,
        x="StoreSize",
        y="WeeklySales",
        palette="Set2",
        ax=ax10
    )
    ax10.set_title("WeeklySales nach StoreSize")
    ax10.set_xlabel("StoreSize")
    ax10.set_ylabel("WeeklySales")
    ax10.set_yscale("log")
    st.pyplot(fig10)

    # ----------------------------------------
    # Ende der deskriptiven Analyse
    # ----------------------------------------
    st.markdown("---")
    st.write(
        "Die deskriptive Analyse ist abgeschlossen. Du kannst die verschiedenen Kennzahlen und Grafiken weiter anpassen oder interaktive Filter hinzufügen.")


page()
