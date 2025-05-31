import pandas as pd
import pulp


# Finde eine optimale Aufteilung von Promotionen auf die einzelnen Departments der Läden für ein bestimmtes Budget, sodass
# der größte Umsatzboost daraus entsteht.
def run_promotion_sales_optimization(df_sales, df_features, df_stores, min_store_size=100_000, budget=50_000,
                                     max_promos_per_week=50,
                                     promo_cost=1_000,
                                     promo_boost=0.10,
                                     ui_status=None):
    # === Daten vorbereiten ===
    update_status(ui_status, "Daten vorbereiten...")
    df_sales["Date"] = pd.to_datetime(df_sales["Date"])
    df_features["Date"] = pd.to_datetime(df_features["Date"])

    df_sales["Week"] = df_sales["Date"].dt.isocalendar().week
    df_sales["Year"] = df_sales["Date"].dt.isocalendar().year

    # Durchschnittlicher Umsatz je Store/Dept/Woche
    weekly_avg = df_sales.groupby(["StoreID", "DeptID", "Year", "Week"])["WeeklySales"].mean().reset_index()

    # Ladengrößenbeschränkung
    eligible_stores = df_stores[df_stores["StoreSize"] > min_store_size]["StoreID"].unique()
    weekly_avg = weekly_avg[weekly_avg["StoreID"].isin(eligible_stores)]

    # Feiertage identifizieren
    df_merged = pd.merge(df_sales, df_features, on=["StoreID", "Date"], how="left")
    df_merged["Week"] = df_merged["Date"].dt.isocalendar().week
    df_merged["Year"] = df_merged["Date"].dt.isocalendar().year
    holiday_weeks = df_merged[df_merged["IsHoliday"] == 1][["StoreID", "Year", "Week"]].drop_duplicates()

    # Nur Nicht-Feiertagswochen behalten
    valid_weeks = pd.merge(
        weekly_avg,
        holiday_weeks,
        on=["StoreID", "Year", "Week"],
        how="left",
        indicator=True
    )
    valid_weeks = valid_weeks[valid_weeks["_merge"] == "left_only"].drop(columns="_merge")

    # === Entscheidungsvariablen ===
    update_status(ui_status, "Optimierungsmodell erstellen...")
    x = pulp.LpVariable.dicts(
        "Promo",
        ((r.StoreID, r.DeptID, r.Year, r.Week) for r in valid_weeks.itertuples(index=False)),
        cat="Binary"
    )

    # Basisumsatz als Dictionary
    base_sales = {
        (r.StoreID, r.DeptID, r.Year, r.Week): r.WeeklySales
        for r in valid_weeks.itertuples(index=False)
    }

    # === Modell definieren ===
    model = pulp.LpProblem("Promotion_Optimization", pulp.LpMaximize)

    # Zielfunktion: Umsatzsteigerung durch Promotion
    model += pulp.lpSum([
        base_sales[key] * promo_boost * x[key]
        for key in x
    ]), "Total_Promotion_Revenue"

    # Nebenbedingung: Budget
    model += pulp.lpSum([
        promo_cost * x[key]
        for key in x
    ]) <= budget, "Budget_Constraint"

    # Nebenbedingung: max. Promotions pro Woche
    unique_weeks = valid_weeks[["Year", "Week"]].drop_duplicates()
    for _, row in unique_weeks.iterrows():
        y, w = row["Year"], row["Week"]
        model += pulp.lpSum([
            x[key]
            for key in x if key[2] == y and key[3] == w
        ]) <= max_promos_per_week, f"Max_Promos_Week_{y}_{w}"

    # === Modell lösen ===
    update_status(ui_status, "Optimierung durchführen...")
    model.solve()

    # === Lösung sammeln ===
    update_status(ui_status, "Lösung wird verarbeitet...")
    solution = [
        {
            "StoreID": s, "DeptID": d, "Year": y, "Week": w,
            "BaseSales": base_sales[(s, d, y, w)],
            "PromoBoostedSales": base_sales[(s, d, y, w)] * (1 + promo_boost)
        }
        for (s, d, y, w), var in x.items() if var.varValue == 1
    ]

    df_solution = pd.DataFrame(solution)

    update_status_state(ui_status, "Optimierung abgeschlossen.", "complete")
    return df_solution, pulp.LpStatus[model.status]


def update_status(ui_status, label):
    update_status_state(ui_status, label, "running")


def update_status_state(ui_status, label, state):
    if ui_status:
        ui_status.update(label=label, state=state)
