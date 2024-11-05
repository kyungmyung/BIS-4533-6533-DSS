import yfinance as f
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib

ticker = 'MSFT'
stock_data = f.download(ticker, start='2020-01-01', end='2024-11-03')
stock_price = stock_data ['Close'].values.reshape(-1, 1)
days = stock_data.index

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(stock_price)

def create_dataset(data, period):
    X, Y = [], []
    for i in range(len(data) - period):
        X.append(data[i:(i + period), 0])
        Y.append(data[i + period, 0])
    return np.array(X), np.array(Y)

period = 60

X,Y = create_dataset(scaled_data, period)
X = X.reshape(X.shape[0], X.shape[1], 1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3 , shuffle=False)

model = Sequential()
model.add(LSTM(10, activation='tanh', return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50, activation='tanh'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)

Y_pred = model.predict(X_test)
Y_test_re = scaler.inverse_transform(Y_test.reshape(-1, 1))
Y_pred_re = scaler.inverse_transform(Y_pred)

plt.figure(figsize=(14, 7))
plt.plot(stock_data.index[len(X_train) + period:], Y_test_re, color='blue', label='Actual Price')
plt.plot(stock_data.index[len(X_train) + period:], Y_pred_re, color='red', label='Predicted Price')
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

test_data = scaled_data[-period:]
test_data = test_data.reshape(1, period, 1)

predicted_price = model.predict(test_data)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"tomorrow's MSFT stock price: {predicted_price[0][0]}")











