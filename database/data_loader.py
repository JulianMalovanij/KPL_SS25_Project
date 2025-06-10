import sqlite3

import pandas as pd
import streamlit as st

from database.data_utils import create_table_map, encode_identifiers, decode_placeholders, trim_forecast_df


@st.cache_data
def load_data(db_path="database/walmart.db"):
    conn = sqlite3.connect(db_path)

    query_sales = "SELECT * FROM WeeklySales"
    query_features = "SELECT * FROM StoreFeature"
    query_stores = "SELECT * FROM Store"

    df_sales = pd.read_sql(query_sales, conn)
    df_features = pd.read_sql(query_features, conn)
    df_stores = pd.read_sql(query_stores, conn)
    conn.close()

    df_sales["Date"] = pd.to_datetime(df_sales["Date"])
    df_features["Date"] = pd.to_datetime(df_features["Date"])

    return df_sales, df_features, df_stores


@st.cache_data
def load_product_data(db_path="database/walmart.db"):
    conn = sqlite3.connect(db_path)

    query_hist = "SELECT * FROM HistoricalDemand"
    query_prod = "SELECT * FROM Product"
    query_cat = "SELECT * FROM ProductCategory"

    df_hist = pd.read_sql(query_hist, conn)
    df_prod = pd.read_sql(query_prod, conn)
    df_cat = pd.read_sql(query_cat, conn)
    conn.close()

    df_hist["Date"] = pd.to_datetime(df_hist["Date"])

    return df_hist, df_prod, df_cat


@st.cache_data
def load_forecast_data(model, identifiers, table_suffix, last_date=None, periods=None,
                       db_path="database/predictions.db"):
    table_map = create_table_map(table_suffix)

    table = table_map.get(model)
    if not table:
        return pd.DataFrame()

    where_clause = " AND ".join([f"{key} = :{key}" for key in identifiers.keys()])
    query = f"SELECT * FROM {table} WHERE {where_clause}"

    # encode identifiers for query (None→placeholder)
    safe_ids = {k: v for k, (_, v) in encode_identifiers(identifiers).items()}

    conn = sqlite3.connect(db_path)
    df_forecast = pd.read_sql(query, conn, params=safe_ids)
    conn.close()

    if "ds" in df_forecast.columns:
        df_forecast["ds"] = pd.to_datetime(df_forecast["ds"])

    # placeholder → None
    return trim_forecast_df(decode_placeholders(df_forecast, identifiers), last_date, periods)


def load_sales_forecast_data(model, store_id, dept_id, last_date=None, periods=None, db_path="database/predictions.db"):
    return load_forecast_data(model, {"StoreID": int(store_id), "DeptID": int(dept_id)}, "_Sales", last_date, periods,
                              db_path)


def load_products_forecast_data(model, wh_code="__NONE__", prod_code="__NONE__", cat_code="__NONE__", last_date=None,
                                periods=None,
                                db_path="database/predictions.db"):
    return load_forecast_data(model, {"ProductCategory": cat_code, "ProductCode": prod_code, "WarehouseCode": wh_code},
                              "_Products", last_date, periods, db_path)


@st.cache_data
def load_full_forecast_data(model, table_suffix, db_path="database/predictions.db"):
    table_map = create_table_map(table_suffix)

    table = table_map.get(model)
    if not table:
        return pd.DataFrame()

    query = f"SELECT * FROM {table}"

    conn = sqlite3.connect(db_path)
    df_forecast = pd.read_sql(query, conn)
    conn.close()

    if "ds" in df_forecast.columns:
        df_forecast["ds"] = pd.to_datetime(df_forecast["ds"])

    return df_forecast


def load_full_sales_forecast_data(model, db_path="database/predictions.db"):
    return load_full_forecast_data(model, "_Sales", db_path)


def load_full_products_forecast_data(model, db_path="database/predictions.db"):
    return load_full_forecast_data(model, "_Products", db_path)


@st.cache_data
def load_multi_forecast_data(models, identifiers, table_suffix, last_date=None, periods=None,
                             db_path="database/predictions.db"):
    table_map = create_table_map(table_suffix)

    where_clause = " AND ".join([f"{key} = :{key}" for key in identifiers.keys()])
    forecasts = {}

    # encode identifiers for query (None→placeholder)
    safe_ids = {k: v for k, (_, v) in encode_identifiers(identifiers).items()}

    conn = sqlite3.connect(db_path)
    for model in models:
        table = table_map.get(model)
        if table:
            query = f"SELECT * FROM {table} WHERE {where_clause}"
            df = pd.read_sql(query, conn, params=safe_ids)
            if "ds" in df.columns:
                df["ds"] = pd.to_datetime(df["ds"])
            forecasts[model] = trim_forecast_df(decode_placeholders(df, identifiers), last_date, periods)
    conn.close()

    return forecasts


def load_multi_sales_forecast_data(models, store_id, last_date=None, periods=None, db_path="database/predictions.db"):
    return load_multi_forecast_data(models, {"StoreID": int(store_id), "DeptID": int(-1)}, "_Sales", last_date, periods,
                                    db_path)


def load_multi_products_forecast_data(models, wh_code="__NONE__", prod_code="__NONE__", cat_code="__NONE__",
                                      last_date=None, periods=None,
                                      db_path="database/predictions.db"):
    return load_multi_forecast_data(models,
                                    {"ProductCategory": cat_code, "ProductCode": prod_code, "WarehouseCode": wh_code},
                                    "_Products", last_date, periods, db_path)
