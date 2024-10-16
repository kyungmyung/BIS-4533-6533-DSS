import pandas as pd

data = pd.read_csv(r'C:\Users\kyungmyung\Downloads\fraud_data.csv')

# Explore the data Structure
# Target Variable
data.columns

# 0: Not Fraud 1: Fraud
y = data['FraudIndicator']
x = data[['AccountBalance','AnomalyScore','TransactionAmount']]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)

import tensorflow as tf
model = tf.keras.models.Sequential()

# Input normalization
X_train = tf.keras.utils.normalize(X_train,axis=0)
X_test = tf.keras.utils.normalize(X_test,axis=0)

model.add(tf.keras.layers.Flatten(input_shape=(3,)))
model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10)







# Check the model performance
from sklearn.metrics import accuracy_score
import numpy as np

y_pred = model.predict(X_test)

# Convert predictions to binary values and flatten the array
y_pred_binary = np.where(y_pred >= 0.5, 1, 0).flatten()

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, y_pred_binary)
print("Accuracy:", accuracy)
