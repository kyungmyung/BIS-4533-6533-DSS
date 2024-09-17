import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

# Part 1
data = pd.read_excel(r'C:\Users\kyungmyung\Downloads\Financial_Data.xlsx')
data = data.fillna(0)

# Select int & float variables
int_columns = data.select_dtypes(include=['int64', 'int32', 'float']).columns
X = data[int_columns]
X = X.drop('Sales/Turnover (Net)', axis=1)

# Target variable to predict!
y = data['Sales/Turnover (Net)']

# RandomForestRegressor to calcualte feature importance
model = RandomForestRegressor(n_estimators=20, random_state=0)
model.fit(X, y)

# feature importance print
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

important_features = importance_df[importance_df['Importance'] >= 0.001]
print(important_features)

# Select three variables that are high scores in the important features
exog_vars = ['Operating Expenses - Total','Revenue - Total','Cost of Goods Sold']
exog = sm.add_constant(data[exog_vars])

model = sm.OLS(data['Sales/Turnover (Net)'], exog)
pooled_regression = model.fit()
pooled_regression.summary()
