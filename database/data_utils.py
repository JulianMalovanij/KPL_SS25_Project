import pandas as pd


def create_table_map(table_suffix):
    return {
        "Prophet": f"ProphetForecast{table_suffix}",
        "ARIMA": f"ArimaForecast{table_suffix}",
        "Holt-Winters": f"HoltWintersForecast{table_suffix}"
    }


def infer_sqlite_type(value):
    """Leitet SQLite-Typ aus Python-Wert ab."""
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "REAL"
    else:
        return "TEXT"


def encode_identifiers(identifiers):
    """
    Für jeden Identifier key→value:
    - ermittelt den SQLite-Typ,
    - ersetzt None durch den passenden Platzhalter.
    Rückgabe: { key: (sql_type, safe_value) }
    """

    def null_fallback(dtype):
        return {
            "INTEGER": -1,
            "REAL": -1.0,
            "TEXT": "__NONE__"
        }[dtype]

    encoded = {}
    for key, value in identifiers.items():
        dtype = infer_sqlite_type(value)
        safe = null_fallback(dtype) if value is None else value
        encoded[key] = (dtype, safe)
    return encoded


def decode_placeholders(df, identifiers):
    """
    Für jede Identifier-Spalte im DataFrame:
    - ersetzt Platzhalter zurück zu None basierend auf original identifiers.
    """
    for key, orig_val in identifiers.items():
        dtype = infer_sqlite_type(orig_val)
        placeholder = {
            "INTEGER": -1,
            "REAL": -1.0,
            "TEXT": "__NONE__"
        }[dtype]
        df[key] = df[key].replace(placeholder, None)
    return df


def trim_forecast_df(forecast_df, last_date, periods):
    """
    Schneidet forecast_df so zu, dass
    - alle Zeilen mit ds <= max(store_df.ds) erhalten bleiben UND
    - von den Zeilen mit ds > max(store_df.ds) nur die ersten `periods` zurückgegeben werden.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame mit Spalte 'ds' (datetime), das historische + prognostizierte Werte enthält.
    last_date : datetime
        letzter Zeitpunkt historischer Daten.
    periods : int
        Anzahl der künftigen Perioden (ds > last_date), die zusätzlich aufgenommen werden.

    Returns
    -------
    pd.DataFrame
        Gefiltertes und nach 'ds' sortiertes DataFrame.
    """
    if periods is None:
        return forecast_df

    # Sicherstellen, dass 'ds' datetime ist
    fc = forecast_df.copy()
    fc['ds'] = pd.to_datetime(fc['ds'])
    fc = fc.sort_values('ds')

    if last_date is None:
        last_date = fc['ds'].iat[0]

    # Teil 1: alle historischen und eventuelle Forecast-Werte bis last_date
    before = fc[fc['ds'] <= last_date]

    # Teil 2: nur die nächsten `periods` Werte nach last_date
    after = fc[fc['ds'] > last_date].iloc[:periods]

    # Zusammenführen und zurückgeben
    result = pd.concat([before, after], ignore_index=True)
    return result.sort_values('ds').reset_index(drop=True)
