#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from datetime import date, timedelta

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Constants
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
RELIANCE_TICKER = "RELIANCE.NS" 

# Streamlit App
st.title('Reliance Stock Forecast App')

st.write(f"Predicting the stock prices of Reliance Industries for the next 30 days.")

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(RELIANCE_TICKER)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Prepare data for forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

# Predict forecast for next 30 days
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

# Date input and forecast button
forecast_date = st.date_input("Enter a date to get the forecasted price", value=date.today() + timedelta(days=1))
if st.button('Forecast'):
    if forecast_date > forecast['ds'].iloc[-1].date():
        st.error("The entered date is beyond the forecast range. Please enter a date within the next 30 days.")
    else:
        forecasted_value = forecast.loc[forecast['ds'] == str(forecast_date), 'yhat'].values[0]
        st.write(f"The forecasted price for {forecast_date} is: ₹{forecasted_value:.2f}")

# Plot the forecast
st.write(f'Forecast plot for the next 30 days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


# In[ ]:




