import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

df = pd.read_csv('stock_prices.csv')

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df = df.dropna()

plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.title('Stock Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

df['Close_Lag1'] = df['Close'].shift(1)
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df.dropna(inplace=True)

model = ARIMA(df['Close'], order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.predict(start=0, end=len(df)-1, dynamic=False)

mae = mean_absolute_error(df['Close'], forecast)
rmse = np.sqrt(mean_squared_error(df['Close'], forecast))
mape = mean_absolute_percentage_error(df['Close'], forecast)

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Actual Close Price')
plt.plot(df.index, forecast, label='ARIMA Forecast', linestyle='--')
plt.title('Stock Price Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
