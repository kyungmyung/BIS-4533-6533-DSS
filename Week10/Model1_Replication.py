import pandas as pd
import tensorflow as tf

data = pd.read_csv(r'C:\Users\kyungmyung\Downloads\fraud_data.csv')

ouput = data['FraudIndicator']
input = data[['AccountBalance' , 'AnomalyScore' , 'TransactionAmount']]

input = tf.keras.utils.normalize(input, axis=0)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(input, ouput, test_size=0.3)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(3,)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)