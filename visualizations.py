import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

        plt.figure(figsize=(14, 7))

        # Aggregiere PromoBoostedSales je Store und Woche
        plot_data = df_solution.groupby(['StoreID', 'YearWeek'])['PromoBoostedSales'].sum().reset_index()

        sns.lineplot(data=plot_data, x='YearWeek', y='PromoBoostedSales', hue='StoreID', marker="o")

        plt.title("Promotion-Boosted Sales pro Store und Woche")
        plt.xlabel("Datum (ISO-Woche)")
        plt.ylabel("Promotion-boosted Umsatz")
        plt.legend(title="StoreID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        return plt
