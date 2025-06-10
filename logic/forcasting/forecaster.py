from datetime import timedelta

import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import database.data_loader as data_loader
from database.data_writer import save_sales_prophet_forecast, save_sales_arima_forecast, save_sales_hw_forecast, \
    save_products_prophet_forecast, save_products_arima_forecast, save_products_hw_forecast


def prophet_forecast(df, periods):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    return forecast


def arima_forecast(df, periods):
    df = df.set_index("ds")
    model = auto_arima(df["y"], seasonal=True, m=52)
    forecast, conf = model.predict(n_periods=periods, return_conf_int=True)
    future_index = pd.date_range(df.index[-1], periods=periods, freq="W")
    return forecast, future_index, conf


def holt_winters_forecast(df, periods):
    df = df.set_index("ds")
    model = ExponentialSmoothing(
        df['y'],
        trend='additive',
        seasonal='additive',
        seasonal_periods=20,
        damped_trend=True
    ).fit()
    forecast = model.forecast(periods)
    future_dates = [df.index.max() + timedelta(weeks=i) for i in range(1, periods + 1)]
    return forecast.values, future_dates


def run_sales_forecast(history, model_option, store_id, dept_id, periods):
    data = None
    if model_option == "Prophet":
        forcast = prophet_forecast(history, periods)
        data = save_sales_prophet_forecast(forcast, store_id, dept_id)
    elif model_option == "ARIMA":
        forecast, future_index, conf = arima_forecast(history, periods)
        data = save_sales_arima_forecast(forecast, future_index, store_id, dept_id, conf_int=conf)
    elif model_option == "Holt-Winters":
        forecast, future_index = holt_winters_forecast(history, periods)
        data = save_sales_hw_forecast(forecast, future_index, store_id, dept_id)

    # Clear cache
    data_loader.load_forecast_data.clear()
    data_loader.load_full_forecast_data.clear()
    data_loader.load_multi_forecast_data.clear()
    return data


def run_products_forecast(history, model_option, periods, wh_code=None, prod_code=None, cat_code=None):
    data = None
    if model_option == "Prophet":
        forcast = prophet_forecast(history, periods)
        data = save_products_prophet_forecast(forcast, wh_code, prod_code, cat_code)
    elif model_option == "ARIMA":
        forecast, future_index, conf = arima_forecast(history, periods)
        data = save_products_arima_forecast(forecast, future_index, wh_code, prod_code, cat_code, conf_int=conf)
    elif model_option == "Holt-Winters":
        forecast, future_index = holt_winters_forecast(history, periods)
        data = save_products_hw_forecast(forecast, future_index, wh_code, prod_code, cat_code)

    # Clear cache
    data_loader.load_forecast_data.clear()
    data_loader.load_full_forecast_data.clear()
    data_loader.load_multi_forecast_data.clear()
    return data


def generate_sales_forecasts(df, periods, model_choices, store_id, dept_id=-1):
    forecasts = {}
    if "Prophet" in model_choices:
        forecasts['Prophet'] = run_sales_forecast(df.copy(), "Prophet", store_id, dept_id, periods)
    if "ARIMA" in model_choices:
        forecasts['ARIMA'] = run_sales_forecast(df.copy(), "ARIMA", store_id, dept_id, periods)
    if "Holt-Winters" in model_choices:
        forecasts['Holt-Winters'] = run_sales_forecast(df.copy(), "Holt-Winters", store_id, dept_id, periods)
    return forecasts
