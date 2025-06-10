import sqlite3

import pandas as pd

from database.data_utils import encode_identifiers


def save_prophet_forecast(forecast, identifiers, table_suffix, db_path="database/predictions.db"):
    # Nur relevante Spalten extrahieren
    columns_to_save = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    if not all(col in forecast.columns for col in columns_to_save):
        raise ValueError(f"Forecast DataFrame muss die Spalten {columns_to_save} enthalten.")

    # Prognose vorbereiten
    df = forecast[columns_to_save].copy()

    # Identifier-Spalten hinzufügen
    for key, value in identifiers.items():
        df[key] = value

    # Schreiben
    write_forecast(df, identifiers, f"ProphetForecast{table_suffix}", db_path)
    return df


def save_sales_prophet_forecast(forecast, store_id, dept_id, db_path="database/predictions.db"):
    return save_prophet_forecast(forecast, {"StoreID": store_id, "DeptID": dept_id},
                                 "_Sales", db_path)


def save_products_prophet_forecast(forecast, wh_code="__NONE__", prod_code="__NONE__", cat_code="__NONE__",
                                   db_path="database/predictions.db"):
    return save_prophet_forecast(forecast,
                                 {"ProductCategory": cat_code, "ProductCode": prod_code, "WarehouseCode": wh_code},
                                 "_Products", db_path)


def save_arima_forecast(predictions, dates, identifiers, table_suffix, db_path="database/predictions.db",
                        conf_int=None):
    if len(predictions) != len(dates):
        raise ValueError("Länge von predictions und dates muss übereinstimmen.")

    # Basis-Vorhersagedaten
    df = pd.DataFrame({
        "ds": pd.to_datetime(dates),
        "yhat": predictions
    })

    if conf_int is not None:
        df["yhat_lower"] = conf_int[:, 0]
        df["yhat_upper"] = conf_int[:, 1]
    else:
        df["yhat_lower"] = None
        df["yhat_upper"] = None

    # Identifier-Spalten hinzufügen
    for key, value in identifiers.items():
        df[key] = value

    # An zentrale Schreibfunktion übergeben
    write_forecast(df, identifiers, f"ArimaForecast{table_suffix}", db_path)
    return df


def save_sales_arima_forecast(predictions, dates, store_id, dept_id, db_path="database/predictions.db", conf_int=None):
    return save_arima_forecast(predictions, dates, {"StoreID": store_id, "DeptID": dept_id},
                               "_Sales", db_path, conf_int)


def save_products_arima_forecast(predictions, dates, wh_code="__NONE__", prod_code="__NONE__", cat_code="__NONE__",
                                 db_path="database/predictions.db", conf_int=None):
    return save_arima_forecast(predictions, dates,
                               {"ProductCategory": cat_code, "ProductCode": prod_code, "WarehouseCode": wh_code},
                               "_Products", db_path, conf_int)


def save_hw_forecast(predictions, dates, identifiers, table_suffix, db_path="database/predictions.db"):
    if len(predictions) != len(dates):
        raise ValueError("Anzahl der Vorhersagen und Zeitstempel muss übereinstimmen.")

    # Basis-Vorhersagedaten
    df = pd.DataFrame({
        "ds": pd.to_datetime(dates),
        "yhat": predictions,
        "yhat_lower": None,
        "yhat_upper": None
    })

    # Identifier-Spalten hinzufügen (z.B. StoreID, DeptID, ProductCode etc.)
    for key, value in identifiers.items():
        df[key] = value

    # An zentrale Schreibfunktion übergeben
    write_forecast(df, identifiers, f"HoltWintersForecast{table_suffix}", db_path)
    return df


def save_sales_hw_forecast(predictions, dates, store_id, dept_id, db_path="database/predictions.db"):
    return save_hw_forecast(predictions, dates, {"StoreID": store_id, "DeptID": dept_id},
                            "_Sales", db_path)


def save_products_hw_forecast(predictions, dates, wh_code="__NONE__", prod_code="__NONE__", cat_code="__NONE__",
                              db_path="database/predictions.db"):
    return save_hw_forecast(predictions, dates,
                            {"ProductCategory": cat_code, "ProductCode": prod_code, "WarehouseCode": wh_code},
                            "_Products", db_path)


def write_forecast(df, identifiers, table_name, db_path="database/predictions.db"):
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d')

    # Schritt 1: Typen ableiten und Platzhalter einsetzen
    safe_identifiers = encode_identifiers(identifiers)

    # Schritt 2: Tabellenschema definieren
    col_defs = [f"{key} {dtype}" for key, (dtype, _) in safe_identifiers.items()]
    col_defs += [
        "ds TEXT",
        "yhat REAL",
        "yhat_lower REAL",
        "yhat_upper REAL"
    ]

    # Schritt 3: Primärschlüssel definieren
    pk_string = ", ".join(list(safe_identifiers.keys()) + ["ds"])

    # Schritt 4: Schreibparameter vorbereiten
    columns = list(safe_identifiers.keys()) + ["ds", "yhat", "yhat_lower", "yhat_upper"]
    col_string = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    update_string = ", ".join(f"{col} = excluded.{col}" for col in ["yhat", "yhat_lower", "yhat_upper"])

    # Schritt 5: Identifier-Spalten zum DataFrame hinzufügen
    for key, (_, value) in safe_identifiers.items():
        df[key] = value

    # Schritt 6: In die Datenbank schreiben
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(col_defs)},
                PRIMARY KEY ({pk_string})
            )
        """)
        insert_sql = f"""
            INSERT INTO {table_name} ({col_string})
            VALUES ({placeholders})
            ON CONFLICT({pk_string}) DO UPDATE SET {update_string}
        """
        conn.executemany(insert_sql, df[columns].itertuples(index=False, name=None))
