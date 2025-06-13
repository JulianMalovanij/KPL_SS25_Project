import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker


def plot_promotion_optimization(df_solution):
    # Sicherstellen, dass Lösung vorhanden ist
    if df_solution.empty:
        print("Keine Promotions in der Lösung.")
        return None
    else:
        # Kombiniere Jahr und Woche zu einer Datumsangabe für Plot
        # Erster Tag der ISO-Woche (Montag)
        df_solution['YearWeek'] = df_solution.apply(
            lambda row1: pd.to_datetime(f"{int(row1['Year'])}-W{int(row1['Week']):02d}-1", format="%G-W%V-%u"),
            axis=1
        )

        # Daten umformen für Gruppierung
        df_melted = df_solution.melt(
            id_vars=['StoreID', 'DeptID', 'YearWeek'],
            value_vars=['BaseSales', 'PromoBoostedSales'],
            var_name='Typ',
            value_name='Sales'
        )

        # Kombinierte Achsenbeschriftung für bessere Gruppierung
        df_melted['Store-Week'] = df_melted['StoreID'].astype(str) + " / " + df_melted['YearWeek'].dt.strftime(
            '%Y-%m-%d')

        # Sortieren nach Datum
        df_melted = df_melted.sort_values(by='YearWeek')

        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.barplot(
            data=df_melted,
            x='Store-Week',
            y='Sales',
            hue='Typ',
            palette=['gray', 'green']
        )

        ax.set_title("Vergleich: BaseSales vs. PromoBoostedSales pro Store und Woche")
        ax.set_xlabel("Store / Woche")
        ax.set_ylabel("Umsatz")
        ax.tick_params(axis='x', labelrotation=90)
        ax.legend(title="Typ")
        ax.set_xmargin(0)
        fig.tight_layout()

        return fig


def prepare_solution_data(df_solution):
    if df_solution.empty:
        return df_solution

    df = df_solution.copy().dropna()
    df['YearWeek'] = df.apply(
        lambda row: pd.to_datetime(f"{int(row['Year'])}-W{int(row['Week']):02d}-1", format="%G-W%V-%u"),
        axis=1
    )
    df = df.sort_values(['StoreID', 'DeptID', 'YearWeek'])
    return df


def plot_sales_boost(df_dept, store_id, dept_id):
    if df_dept.empty:
        return None

    # Zeitstempel in ISO-Wochen-String
    df_dept = df_dept.copy()
    df_dept['YearWeekStr'] = df_dept['YearWeek'].dt.strftime('%G-W%V')

    # Kategorie als sortierte Achse
    df_dept['Store-Week'] = pd.Categorical(
        df_dept['YearWeekStr'],
        categories=sorted(df_dept['YearWeekStr'].unique()),
        ordered=True
    )

    # Umsatzdaten vorbereiten
    df_melted = df_dept.melt(
        id_vars=['Store-Week'],
        value_vars=['NetGain', 'PotentialGain'],
        var_name='Typ',
        value_name='Sales'
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=df_melted,
        x='Store-Week',
        y='Sales',
        hue='Typ',
        palette=['green', 'gray'],
        ax=ax
    )

    # Zweite Achse: Boost-Faktoren
    ax2 = ax.twinx()
    boost = df_dept[['Store-Week', 'PotentialEffectiveBoost', 'BoostPotential', 'Promotion']].copy()

    # Linienplot der Boost-Werte
    ax2.plot(boost['Store-Week'], boost['PotentialEffectiveBoost'], color='blue', marker='o',
             label='effektives Potential von Promotionen')
    ax2.plot(boost['Store-Week'], boost['BoostPotential'], color='purple', marker='o',
             label='Effektivität von Promotionen')
    ax2.set_ylabel("Boost-Faktor", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, max(1, boost[['PotentialEffectiveBoost', 'BoostPotential']].max().max() * 1.2))
    ax2.set_xmargin(0)

    # Rote Marker für Promotionen (x==1)
    promo_points = boost[boost['Promotion'] == 1]
    for i, row in promo_points.iterrows():
        x_pos = row['Store-Week']
        y_pos = df_dept[df_dept['Store-Week'] == x_pos]['NetGain'].values
        if len(y_pos) > 0:
            ax.plot(x_pos, y_pos[0] * 1.05, 'ro', markersize=7)

    # X-Achse lesbar machen
    ax.set_xlabel("Jahr / Kalenderwoche")
    ax.set_ylabel("Wirkung der Promotionen")
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xmargin(0)
    ax.set_ylim(0, df_melted['Sales'].max() * 1.2)

    # Ticks automatisch anpassen, bei zu vielen Wochen → reduzierte Anzeige
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15, prune='both'))

    # Titel und Legenden
    ax.set_title(f"Store {store_id} – Dept {dept_id}")
    ax.legend(title="Nettogewinn (Umsatzsteigerung - Promotionskosten)", loc='upper left',
              labels=['Verwirklichter Nettogewinn', 'Potentieller Nettogewinn'])
    ax2.legend(loc='upper right')

    fig.tight_layout()
    return fig
