import multiprocessing
import traceback
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pulp

from logic.optimization.helper import report_status, start_ui_status_updater, create_solver

# Konstante Big-M (sollte größer als maximaler Boost sein)
M = 2


def run_single_store_dept_optimization(args):
    store_id, dept_id, df_sales, df_features, params, ui_status = args

    # Filter nur auf aktuellen Store/Dept
    filtered_sales = df_sales[(df_sales["StoreID"] == store_id) & (df_sales["DeptID"] == dept_id)].copy()

    try:
        return run_promotion_sales_optimization(
            filtered_sales,
            df_features,
            boost_max=params["boost_max"],
            decay_factor=params["decay_factor"],
            recovery_rate=params["recovery_rate"],
            cost_rate=params["cost_rate"],
            store_id=store_id,
            dept_id=dept_id,
            ui_status=ui_status,
            solver_timeout=params["solver_timeout"],
            parallel=params["parallel"]
        )
    except Exception as e:
        traceback.print_exc()
        report_status(ui_status, store_id, dept_id, f"Fehler aufgetreten: {e}.", "error")
        return pd.DataFrame([create_data_row(store_id, dept_id)]), f"Error for Store {store_id} Dept {dept_id}: {e}"


def run_promotion_sales_optimization_all(df_sales, df_features,
                                         boost_max=0.15,
                                         decay_factor=0.5,
                                         recovery_rate=0.05,
                                         cost_rate=0.05,
                                         ui_status=None,
                                         parallel=True,
                                         solver_timeout=150):
    # Queue für Statusupdates
    manager = multiprocessing.Manager()
    status_queue = manager.Queue()

    unique_pairs = df_sales[["StoreID", "DeptID"]].drop_duplicates().dropna().values.tolist()
    total = len(unique_pairs)
    # Vermeide unnötigen Overhead durch multiprocessing bei kleinen Problemgrößen
    parallel = parallel and total > multiprocessing.cpu_count() * 0.5
    params = {
        "boost_max": boost_max,
        "decay_factor": decay_factor,
        "recovery_rate": recovery_rate,
        "cost_rate": cost_rate,
        "solver_timeout": solver_timeout,
        "parallel": not parallel,
    }

    args = [(store_id, dept_id, df_sales, df_features, params, status_queue) for store_id, dept_id in
            unique_pairs]

    stop_event = start_ui_status_updater(ui_status, status_queue, total)

    if parallel:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(run_single_store_dept_optimization, args)
    else:
        results = [run_single_store_dept_optimization(arg) for arg in args]

    solutions = [df for df, status in results if not df.empty]
    combined_result = pd.concat(solutions, ignore_index=True) if solutions else pd.DataFrame()

    # Stoppe den Updater-Thread
    stop_event.set()

    return combined_result, results


def run_promotion_sales_optimization(df_sales, df_features,
                                     boost_max=0.15,
                                     decay_factor=0.5,
                                     recovery_rate=0.05,
                                     cost_rate=0.05,
                                     solver_timeout=150,
                                     store_id=None,
                                     dept_id=None,
                                     ui_status=None,
                                     parallel=False):
    if cost_rate >= boost_max:
        report_status(ui_status, store_id, dept_id,
                      f"Es können keine Promotionen durchgeführt werden, wenn die Kosten ({cost_rate:.2f}) höher als der maximale Zuwachs ({boost_max:.2f}) sind.",
                      "error")
        return pd.DataFrame([create_data_row(store_id, dept_id)]), pulp.const.LpStatus[pulp.const.LpStatusInfeasible]

    report_status(ui_status, store_id, dept_id, "Daten vorbereiten...")

    # Zeitspalten
    df_sales["Week"] = df_sales["Date"].dt.isocalendar().week.astype(int)
    df_sales["Year"] = df_sales["Date"].dt.isocalendar().year.astype(int)

    # Durchschnittsumsatz pro Woche/Store/Dept
    weekly_sales_raw = df_sales.groupby(["StoreID", "DeptID", "Year", "Week"])["WeeklySales"].mean().reset_index()

    # Ausreißerentfernung
    weekly_sales = weekly_sales_raw.groupby(["StoreID", "DeptID"], group_keys=False).apply(remove_outliers).reset_index(
        drop=True)

    # Feiertage entfernen
    df_merged = pd.merge(
        df_sales.drop(columns=["IsHoliday"], errors="ignore"),
        df_features,
        on=["StoreID", "Date"],
        how="left"
    )
    df_merged["Week"] = df_merged["Date"].dt.isocalendar().week.astype(int)
    df_merged["Year"] = df_merged["Date"].dt.isocalendar().year.astype(int)
    holiday_weeks = df_merged[df_merged["IsHoliday"] == 1][["StoreID", "Year", "Week"]].drop_duplicates()

    weekly_sales = pd.merge(
        weekly_sales,
        holiday_weeks,
        on=["StoreID", "Year", "Week"],
        how="left",
        indicator=True
    )
    weekly_sales = weekly_sales[weekly_sales["_merge"] == "left_only"].drop(columns="_merge")

    # Zeilen mit NA in Schlüsselspalten sicher entfernen
    weekly_sales = weekly_sales.dropna(subset=["StoreID", "DeptID", "Year", "Week", "WeeklySales"])

    report_status(ui_status, store_id, dept_id, "Entscheidungsvariablen erstellen...")

    # Keys erzeugen ohne NA
    week_tuples = [
        (r.StoreID, r.DeptID, r.Year, r.Week)
        for r in weekly_sales.itertuples(index=False)
        if pd.notna(r.StoreID) and pd.notna(r.DeptID) and pd.notna(r.Year) and pd.notna(r.Week)
    ]

    # Basisumsatz-Dictionary
    base_sales = {
        key: val
        for key, val in zip(week_tuples, weekly_sales["WeeklySales"])
    }

    # Boost-Potenzial nach Umsatzstärke
    boost_potential = compute_boost_potential(base_sales, fallback_value=0.0, normalize=True)
    promo_cost = {key: base_sales[key] * cost_rate for key in base_sales}

    # Entscheidungsvariablen
    # x ist unser Ziel. 0 → Keine Promo in der Woche, 1 → Promo!
    x = pulp.LpVariable.dicts("Promo", week_tuples, cat="Binary")
    # dynamic_boost wird für die dynamische Anpassung des Umsatzboosts einer Promo benötigt, welcher jeweils darauf
    # basiert, ob in der Vorwoche eine Promotion stattgefunden hat (--> Decay) oder nicht (--> Recovery)
    dynamic_boost = pulp.LpVariable.dicts("DynamicBoost", week_tuples, lowBound=0, upBound=boost_max)
    # dynamic_boost_effective ist für die linearisierung der Zielfunktion nötig. In dieser müssten ansonsten die
    # zwei Variablen x und dynamic_boost multipliziert werden, was ein nichtlinieares Problem ist.
    dynamic_boost_effective = pulp.LpVariable.dicts("DynamicBoostEffective", week_tuples, lowBound=0, upBound=boost_max)

    # Hilfsvariablen (global abrufbar für Debug-Zwecke)
    # limited_recovery modeliert min(recovery_rate hoch blah, boost_max) um OOB zu vermeiden
    limited_recovery = pulp.LpVariable.dicts(f"LimitedRecovery", week_tuples, lowBound=0, upBound=boost_max)
    # z_decay und z_recovery modellieren mittels Big-M dynamic_boost[prev] * x[prev] für decay und recovery des boosts
    z_decay = pulp.LpVariable.dicts(f"Z_Decay", week_tuples, lowBound=0)
    z_recovery = pulp.LpVariable.dicts(f"Z_Recovery", week_tuples, lowBound=0, upBound=boost_max)

    report_status(ui_status, store_id, dept_id, "Triviale Initiallösung erstellen...")
    sorted_keys = sorted(week_tuples, key=lambda t: (t[0], t[1], t[2], t[3]))
    # Versuche, die triviale Lösung ("Keine Promotionen") als Startpunkt zu verwenden → Ziel-Fkt = 0
    for key in sorted_keys:
        x[key].setInitialValue(0)

    report_status(ui_status, store_id, dept_id, "Modell definieren...")

    model = pulp.LpProblem("Adaptive_Promotion_Model", pulp.LpMaximize)

    # Zielfunktion: Netto-Umsatz (Boost - Promo-Kosten)
    model += pulp.lpSum([
        base_sales[key] * dynamic_boost_effective[key] * boost_potential[key] - promo_cost[key] * x[key]
        for key in sorted_keys
    ]), "Net_Promotion_Benefit"

    # Dynamik (Decay / Recovery)
    for i, key in enumerate(sorted_keys):
        prev, gap_weeks = get_latest_previous_week(key, set(sorted_keys))
        if prev:
            # Berechne erwarteten Boost nach Recovery über mehrere Wochen
            # angenommen: keine Promo während der Lücke (x[prev] = 0 für alle)
            recovery_multiplier = (1 + recovery_rate) ** gap_weeks

            # Begrenze die maximale Recovery auf boost_max
            # Binary-Variable zur Auswahl, ob capped
            is_capped = pulp.LpVariable(f"IsCapped_{key}", cat="Binary")

            # McCormick-artige Hülle für: limited_recovery = min(dynamic_boost[prev] * recovery_multiplier, boost_max)
            model += limited_recovery[key] <= dynamic_boost[prev] * recovery_multiplier, f"RecoveryMin_UB1_{key}"
            model += limited_recovery[key] <= boost_max, f"RecoveryMin_UB2_{key}"
            model += limited_recovery[key] >= dynamic_boost[
                prev] * recovery_multiplier - M * is_capped, f"RecoveryMin_LB1_{key}"
            model += limited_recovery[key] >= boost_max - M * (1 - is_capped), f"RecoveryMin_LB2_{key}"

            # Da 2 Variablen multipliziert (dynamic_boost[prev]*x[prev] → Linearisierung mit Big-M Constraints
            # Hilfsvariablen für beide Fälle (--> verschoben für Debug-Zwecke)
            # z_decay und z_recovery

            # Fall 1: x == 1 → Decay
            model += z_decay[key] <= dynamic_boost[prev], f"Z_Decay_Upper_{key}"
            model += z_decay[key] <= boost_max * x[prev], f"Z_Decay_Upper_Max_{key}"
            model += z_decay[key] >= dynamic_boost[prev] - M * (1 - x[prev]), f"Z_Decay_Lower_{key}"

            # Fall 2: x == 0 → Recovery
            model += z_recovery[key] <= limited_recovery[key], f"Z_Recovery_Upper_{key}"
            model += z_recovery[key] <= boost_max * (1 - x[prev]), f"Z_Recovery_Upper_Max_{key}"
            model += z_recovery[key] >= (limited_recovery[key] - M * x[prev]), f"Z_Recovery_Lower_{key}"

            # Gesamtwert: entweder Decay oder Recovery
            model += dynamic_boost[key] == z_decay[key] * decay_factor + z_recovery[key], f"Boost_Update_{key}"

        else:
            # Startwert
            model += dynamic_boost[key] == boost_max

        # Lineare Bedingungen zur Modellierung von boost_effective = dynamic_boost * x
        # maximal dynamic_boost
        model += dynamic_boost_effective[key] <= dynamic_boost[key], f"BoostEff_LE_Boost_{key}"
        # maximal boost_max oder 0 (falls x = 0)
        model += dynamic_boost_effective[key] <= boost_max * x[key], f"BoostEff_LE_IfX_{key}"
        # minimal dynamic_boost oder 0 (falls x = 0, da var. als >= 0 definiert)
        model += (dynamic_boost_effective[key] >= dynamic_boost[key] - boost_max * (1 - x[key]),
                  f"BoostEff_GE_BoostIfX_{key}")

    report_status(ui_status, store_id, dept_id, "Optimierung wird durchgeführt...")
    solver = create_solver(solver_timeout=solver_timeout, multithreading=parallel)
    model.solve(solver)

    report_status(ui_status, store_id, dept_id, "Lösung verarbeiten...")

    # Ergebnisse extrahieren
    result = [
        create_data_row(store_id=key[0], dept_id=key[1], year=key[2], week=key[3], x=x[key].varValue,
                        base_sales=base_sales[key], dynamic_boost=dynamic_boost[key].varValue,
                        boost_potential=boost_potential[key], promo_cost=promo_cost[key])
        for key in sorted_keys
    ]

    df_solution = pd.DataFrame(result)
    report_status(ui_status, store_id, dept_id, "Optimierung abgeschlossen.", "complete")
    return df_solution, pulp.LpStatus[model.status]


# IQR Methode zur Ausreißerentfernung
def remove_outliers(df):
    series = df["WeeklySales"]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(series >= lower) & (series <= upper)]


# Berechnet die Boost-Potenziale je Store/Dept/Year/Week basierend auf dem log-inversen Umsatz.
def compute_boost_potential(base_sales, fallback_value=0.0, normalize=False):
    boost_potential = {}

    for key, value in base_sales.items():
        if value > 0:
            boost = 1 / np.log1p(value)
        else:
            boost = fallback_value
        boost_potential[key] = boost

    if normalize:
        values = np.array(list(boost_potential.values()))
        min_val, max_val = np.min(values), np.max(values)
        if max_val > min_val:
            for key in boost_potential:
                boost_potential[key] = (boost_potential[key] - min_val) / (max_val - min_val)
        else:
            # Alle Werte gleich → alles auf 0 setzen
            boost_potential = {key: 0.0 for key in boost_potential}

    return boost_potential


def get_latest_previous_week(key, available_keys):
    s, d, y, w = key
    current_date = date.fromisocalendar(y, w, 1)  # Montag der Woche

    # Suche rückwärts nach der nächsten verfügbaren Woche im Datensatz
    for delta_weeks in range(1, 53):  # maximal 1 Jahr zurück
        prev_date = current_date - timedelta(weeks=delta_weeks)
        prev_key = (s, d, prev_date.isocalendar().year, prev_date.isocalendar().week)
        if prev_key in available_keys:
            return prev_key, delta_weeks
    return None, None  # kein gültiger Vorgänger


# Methode die immer eine definierte Row zurückgibt, auch wenn einige Teile davon nicht berechnet werden können
def create_data_row(store_id, dept_id, year=None, week=None, x=None, base_sales=None, dynamic_boost=None,
                    boost_potential=None, promo_cost=None):
    def safe_multiply(*args):
        try:
            if any(a is None for a in args):
                return None
            result = 1
            for a in args:
                result *= a
            return result
        except Exception:
            return None

    def safe_add(a, b):
        try:
            if a is None or b is None:
                return None
            return a + b
        except Exception:
            return None

    pot_effective_boost = safe_multiply(dynamic_boost, boost_potential)
    pot_net_gain = safe_multiply(base_sales, dynamic_boost, boost_potential)
    if pot_net_gain is not None and promo_cost is not None:
        pot_net_gain = pot_net_gain - promo_cost
    else:
        pot_net_gain = None

    boosted_sales = None
    if base_sales is not None and pot_effective_boost is not None:
        boosted_sales = safe_multiply(base_sales, safe_add(1, pot_effective_boost))

    return {
        "StoreID": store_id,
        "DeptID": dept_id,
        "Year": year,
        "Week": week,
        "Promotion": x,
        "BaseSales": base_sales,
        "BoostFactor": dynamic_boost,
        "BoostPotential": boost_potential,
        "EffectiveBoost": safe_multiply(pot_effective_boost, x),
        "PotentialEffectiveBoost": pot_effective_boost,
        "PromoBoostedSales": safe_multiply(boosted_sales, x),
        "PotentialPromoBoostedSales": boosted_sales,
        "PromoCost": promo_cost,
        "NetGain": safe_multiply(pot_net_gain, x),
        "PotentialGain": pot_net_gain
    }
