# forecast_dept_level.py
# Prognose pro Artikel (DeptID) 

import pandas as pd
import sqlite3
from prophet import Prophet
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np

DB_PATH = "walmart.db"

def load_aggregated_data(dept_id):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT Date, SUM(WeeklySales) as Sales
        FROM WeeklySales
        WHERE DeptID = {dept_id}
        GROUP BY Date
        ORDER BY Date;
    """
    df = pd.read_sql(query, conn, parse_dates=['Date'])
    df.set_index("Date", inplace=True)
    conn.close()
    return df

def prophet_forecast(dept_id):
    df = load_aggregated_data(dept_id).reset_index()
    df = df.rename(columns={"Date": "ds", "Sales": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=104, freq='W')
    forecast = model.predict(future)
    return forecast.set_index("ds")["yhat"]

def arima_forecast(dept_id):
    df = load_aggregated_data(dept_id)
    model = auto_arima(df, seasonal=True, m=52)
    forecast = model.predict(n_periods=104)
    future_index = pd.date_range(df.index[-1], periods=104, freq="W")
    return pd.Series(forecast, index=future_index)

def lstm_forecast(dept_id):
    df = load_aggregated_data(dept_id)
    data = df.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    look_back = 12
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:i + look_back])
        y.append(data_scaled[i + look_back])

    X, y = np.array(X), np.array(y)
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=10, batch_size=8, verbose=0)

    input_seq = data_scaled[-look_back:]
    forecasts = []
    for _ in range(104):
        input_reshaped = input_seq.reshape((1, look_back, 1))
        next_val = model.predict(input_reshaped, verbose=0)
        forecasts.append(next_val[0][0])
        input_seq = np.append(input_seq[1:], [[next_val]], axis=0)

    forecast_rescaled = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    future_index = pd.date_range(df.index[-1], periods=104, freq="W")
    return pd.Series(forecast_rescaled, index=future_index)
