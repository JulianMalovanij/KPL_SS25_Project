import sqlite3

import pandas as pd
import streamlit as st


@st.cache_data
def load_data(db_path="walmart.db"):
    conn = sqlite3.connect(db_path)

    query_sales = "SELECT StoreID, DeptID, Date, WeeklySales FROM WeeklySales"
    query_features = "SELECT StoreID, Date, IsHoliday FROM StoreFeature"
    query_stores = "SELECT StoreID, StoreSize FROM Store"

    df_sales = pd.read_sql(query_sales, conn)
    df_features = pd.read_sql(query_features, conn)
    df_stores = pd.read_sql(query_stores, conn)
    conn.close()

    df_sales["Date"] = pd.to_datetime(df_sales["Date"])
    df_features["Date"] = pd.to_datetime(df_features["Date"])

    return df_sales, df_features, df_stores
