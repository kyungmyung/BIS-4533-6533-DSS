import yfinance as f
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
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

period = 30
X,Y = create_dataset(scaled_data, period)
X = X.reshape(X.shape[0], X.shape[1], 1)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3 , shuffle=False)



model = Sequential()
model.add(SimpleRNN(10, activation='tanh', input_shape=(X.shape[1], 1), return_sequences=True))
model.add(SimpleRNN(50, activation='tanh'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)


predictions = []
for i in range(period, len(scaled_data)):
    test_data = scaled_data[i - period:i].reshape(1, period, 1)
    predicted_price = model.predict(test_data)
    predictions.append(predicted_price[0, 0])


predictions = np.array(predictions).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predictions)

actual_prices = stock_price[period:]
actual_dates = days[period:]


plt.figure(figsize=(12, 6))
plt.plot(actual_dates, actual_prices, label='Actual Price', color='blue')
plt.plot(actual_dates, predicted_prices, label='Predicted Price', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.legend()
plt.show()


test_data = scaled_data[-period:]
test_data = test_data.reshape(1, period, 1)

predicted_price = model.predict(test_data)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"tomorrow's MSFT stock price: {predicted_price[0][0]}")











