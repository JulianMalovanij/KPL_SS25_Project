import sqlite3

import pandas as pd
import streamlit as st


@st.cache_data
def load_data(db_path="walmart.db"):
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
def load_forecast_data(model, store_id, dept_id, db_path="predictions.db"):
    query = None
    if model == "Prophet":
        query = "SELECT ds, yhat, yhat_lower, yhat_upper FROM ProphetForecast WHERE StoreID = :store_id AND DeptID = :dept_id"
    if model == "ARIMA":
        query = "SELECT ds, yhat, yhat_lower, yhat_upper FROM ArimaForecast WHERE StoreID = :store_id AND DeptID = :dept_id"
    if model == "LSTM":
        query = "SELECT ds, yhat FROM LstmForecast WHERE StoreID = :store_id AND DeptID = :dept_id"

    if query is None:
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    df_forecast = pd.read_sql(query, conn, params={"store_id": int(store_id), "dept_id": int(dept_id)})
    conn.close()

    df_forecast["ds"] = pd.to_datetime(df_forecast["ds"])

    return df_forecast
