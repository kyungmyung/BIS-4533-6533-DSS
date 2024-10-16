import pandas as pd
from functools import reduce

# account data load - Bank
acc = pd.read_csv(r'C:\Users\kyungmyung\Downloads\data\account_activity.csv')
# Customer info
cus = pd.read_csv(r'C:\Users\kyungmyung\Downloads\data\customer_data.csv')
# Fraud info
fraud = pd.read_csv(r'C:\Users\kyungmyung\Downloads\data\fraud_indicators.csv')
# Suspicious  activity
sus = pd.read_csv(r'C:\Users\kyungmyung\Downloads\data\suspicious_activity.csv')
# Purchase activity
pur = pd.read_csv(r'C:\Users\kyungmyung\Downloads\data\merchant_data.csv')
# Transaction category
tran = pd.read_csv(r'C:\Users\kyungmyung\Downloads\data\transaction_category_labels.csv')
# Amount of Purchase
amo = pd.read_csv(r'C:\Users\kyungmyung\Downloads\data\amount_data.csv')
# Anonymous Score
anony = pd.read_csv(r'C:\Users\kyungmyung\Downloads\data\anomaly_scores.csv')
# Transaction Metadata
tran_meta = pd.read_csv(r'C:\Users\kyungmyung\Downloads\data\transaction_metadata.csv')
# Transaction Record
tran_record = pd.read_csv(r'C:\Users\kyungmyung\Downloads\data\transaction_records.csv')

# Customer meta info merging
dfs = [cus, acc, sus]
cus_data = reduce(lambda left, right: pd.merge(left, right, on='CustomerID'), dfs)

# Label for transaction
dfs = [fraud, tran, amo, anony, tran_meta, tran_record]
tran_df = reduce(lambda left, right: pd.merge(left, right, on='TransactionID'), dfs)

# Combine customer data with transaction data
data = pd.merge(cus_data, tran_df, on='CustomerID')

# Explore the data Structure
# Target Variable
data.columns

# 0: Not Fraud 1: Fraud
y = data['FraudIndicator']
x = data[['AccountBalance','AnomalyScore','TransactionAmount']]

# Plotting Y
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
sns.countplot(x=y )

plt.title('Count Plot for Fraud')
plt.xlabel('Fraud')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels if they are long
plt.show()

plt.close()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)

import tensorflow as tf
model = tf.keras.models.Sequential()

# Input normalization
X_train = tf.keras.utils.normalize(X_train,axis=0)
X_test = tf.keras.utils.normalize(X_test,axis=0)

model.add(tf.keras.layers.Flatten(input_shape=(3,)))
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
