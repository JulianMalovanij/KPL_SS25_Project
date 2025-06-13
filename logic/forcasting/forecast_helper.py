# Zusatzfunktionen für KPIs
import pandas as pd


def calculate_kpis(df):
    total = df['y'].sum()
    weekly_avg = df.set_index('ds').resample('W').sum().mean().values[0]
    volatility = df['y'].std()
    growth = ((df['y'].iloc[-1] - df['y'].iloc[0]) / df['y'].iloc[0]) * 100 if df['y'].iloc[0] != 0 else 0
    return total, weekly_avg, volatility, growth


# Verfügbare Kombinationen laden
def get_available_combinations(df_hist, df_prod, df_cat):
    df_combos = df_hist[["ProductCode", "WarehouseCode", "ProductCategory"]].drop_duplicates().sort_values(
        ["ProductCode", "WarehouseCode"]).reset_index(drop=True)
    df_products = df_prod["ProductCode"].sort_values().reset_index(drop=True)
    df_categories = df_cat["Name"].sort_values().reset_index(drop=True)
    df_cat_lager = df_hist[["ProductCategory", "WarehouseCode"]].drop_duplicates().sort_values(
        ["ProductCategory", "WarehouseCode"]).reset_index(drop=True)
    return df_combos, df_products.tolist(), df_categories.tolist(), df_cat_lager


def prepare_product_data(df_hist, identifiers):
    """
    Filtert df_hist nach allen in `identifiers` angegebenen Spaltenwerten
    und bereitet daraus eine Zeitreihe (ds, y) aggregiert per Tag vor.

    identifiers: Dict[str, Any], z.B. {"ProductCode": "P1", "WarehouseCode": "W1"}
    """

    # 1) Kopie und Filtern nach allen Identifier-Paaren (None ignorieren)
    df = df_hist.copy()
    for key, value in identifiers.items():
        if value is not None:
            df = df[df[key] == value]

    # 2) Auf relevante Spalten reduzieren und sortieren
    df = (
        df[["Date", "OrderDemand"]]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # 3) Spalten umbenennen und Demand bereinigen
    df = df.rename(columns={"Date": "ds", "OrderDemand": "y"})
    df["y"] = (
        df["y"]
        .astype(str)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df = df.dropna(subset=["y"])

    # 4) Bei mehrfachen Einträgen pro Tag summieren
    return df.groupby("ds", as_index=False)["y"].sum()
