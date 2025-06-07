import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
from keras.models import Sequential
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

from data_writer import save_prophet_forecast, save_arima_forecast, save_lstm_forecast


def prophet_forecast(df):
    df = df.rename(columns={"Date": "ds", "WeeklySales": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=104, freq='W')
    forecast = model.predict(future)
    return forecast


def arima_forecast(df):
    df = df.set_index("Date")
    model = auto_arima(df["WeeklySales"], seasonal=True, m=52)
    forecast, conf = model.predict(n_periods=104, return_conf_int=True)
    future_index = pd.date_range(df.index[-1], periods=104, freq="W")
    return forecast, future_index, conf


def lstm_forecast(df):
    df = df.set_index("Date")
    data = df["WeeklySales"].values.reshape(-1, 1)
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
        next_val_float = next_val[0][0]  # Float-Wert extrahieren
        forecasts.append(next_val_float)

        # Richtiges Append: 2D-Array mit Shape (1,1)
        next_val_reshaped = np.array([[next_val_float]])
        input_seq = np.append(input_seq[1:], next_val_reshaped, axis=0)

    forecast_rescaled = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    future_index = pd.date_range(df.index[-1], periods=104, freq="W")
    return forecast_rescaled, future_index


def run_forecast(history, model_option, store_id, dept_id):
    if model_option == "Prophet":
        forcast = prophet_forecast(history)
        save_prophet_forecast(forcast, store_id, dept_id)
    elif model_option == "ARIMA":
        forecast, future_index, conf = arima_forecast(history)
        save_arima_forecast(forecast, future_index, store_id, dept_id, conf_int=conf)
    elif model_option == "LSTM":
        forecast, future_index = lstm_forecast(history)
        save_lstm_forecast(forecast, future_index, store_id, dept_id)
