import pandas as pd

data = pd.read_excel(r'C:\Users\kyungmyung\Downloads\Financial_Data.xlsx')
data.columns

# Select Numbers only
data = data.select_dtypes(include=['int64', 'float64'])
# Null data clean
data = data.dropna(subset=['Sales/Turnover (Net)']).reset_index(drop=True)
# Replace input features
data = data.fillna(0)

# y is sales.
# input features:x are
y = data['Sales/Turnover (Net)']
x = data.drop('Sales/Turnover (Net)', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3 , shuffle=False)

import tensorflow as tf
model = tf.keras.models.Sequential()

# Input normalization
X_train = tf.keras.utils.normalize(X_train,axis=0)
X_test = tf.keras.utils.normalize(X_test,axis=0)

# Output normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
Y_train = scaler.fit_transform(Y_train.values.reshape(-1, 1))
Y_test = scaler.transform(Y_test.values.reshape(-1, 1))

# Deep NL Structure
model.add(tf.keras.layers.Flatten(input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=10)

# Make predictions
predictions = model.predict(X_test)

# Original Value Convert
predictions_original_scale = scaler.inverse_transform(predictions)

# Original data convert
Y_test_original_scale = scaler.inverse_transform(Y_test)

print(predictions_original_scale)

import matplotlib.pyplot as plt

n_samples = 30
Y_test_last = Y_test_original_scale[-n_samples:]
predictions_last = predictions_original_scale[-n_samples:]

plt.figure(figsize=(10,6))

plt.plot(Y_test_last.flatten(), label='Actual Sales', color='blue', linestyle='-', marker='o', markersize=1)

plt.plot(predictions_last.flatten(), label='Predicted Sales', color='red', linestyle='--', marker='x', markersize=1)

plt.title('Actual vs Predicted Sales/Turnover (Net) - Last 30 Samples')
plt.xlabel('Samples')
plt.ylabel('Sales/Turnover (Net)')
plt.legend()

plt.show()


