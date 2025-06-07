import sqlite3

import numpy as np
import pandas as pd


def save_prophet_forecast(forecast, store_id, dept_id, db_path="predictions.db"):
    # Nur relevante Spalten extrahieren
    columns_to_save = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    if not all(col in forecast.columns for col in columns_to_save):
        raise ValueError(f"Forecast DataFrame muss die Spalten {columns_to_save} enthalten.")

    forecast = forecast[columns_to_save].copy()
    forecast['StoreId'] = store_id
    forecast['DepartmentId'] = dept_id

    # Spaltenreihenfolge
    forecast = forecast[['StoreId', 'DepartmentId', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    write_forecast(forecast, "ProphetForecast", db_path)


def save_arima_forecast(predictions, dates, store_id, dept_id, conf_int=None,
                        db_path="predictions.db"):
    if len(predictions) != len(dates):
        raise ValueError("Länge von predictions und dates muss übereinstimmen.")

    df = pd.DataFrame({
        'ds': pd.to_datetime(dates),
        'yhat': predictions,
        'StoreId': store_id,
        'DepartmentId': dept_id
    })

    if conf_int is not None:
        df['yhat_lower'] = conf_int[:, 0]
        df['yhat_upper'] = conf_int[:, 1]
    else:
        df['yhat_lower'] = None
        df['yhat_upper'] = None

    df = df[['StoreId', 'DepartmentId', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Datenbank schreiben
    write_forecast(df, "ArimaForecast", db_path)


def save_lstm_forecast(predictions, dates, store_id, dept_id, db_path="predictions.db"):
    # Korrigieren, falls 2D-Ausgabe: (n,1) → (n,)
    if isinstance(predictions, np.ndarray) and predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions.flatten()

    if len(predictions) != len(dates):
        raise ValueError("Anzahl der Vorhersagen und Zeitstempel muss übereinstimmen.")

    # DataFrame im Ziel-Schema erstellen
    df = pd.DataFrame({
        'StoreId': store_id,
        'DeptId': dept_id,
        'ds': pd.to_datetime(dates),
        'yhat': predictions,
        'yhat_lower': None,
        'yhat_upper': None
    })

    # Übergabe an zentrale SQL-Funktion
    write_forecast(df, "LstmForecast", db_path)


def write_forecast(df, table_name, db_path="predictions.db"):
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d')

    # Datenbank schreiben
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    StoreID INTEGER NOT NULL,
                    DeptID INTEGER NOT NULL,
                    ds TEXT NOT NULL,
                    yhat REAL,
                    yhat_lower REAL,
                    yhat_upper REAL,
                    PRIMARY KEY (StoreID, DeptID, ds)
                )
            """)

        data = list(df.itertuples(index=False, name=None))

        insert_sql = f"""
                INSERT INTO {table_name}
                (StoreID, DeptID, ds, yhat, yhat_lower, yhat_upper)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(StoreId, DeptID, ds)
                DO UPDATE SET
                    yhat = excluded.yhat,
                    yhat_lower = excluded.yhat_lower,
                    yhat_upper = excluded.yhat_upper
            """

        conn.executemany(insert_sql, data)
