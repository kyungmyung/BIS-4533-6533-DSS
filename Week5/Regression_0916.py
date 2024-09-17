import pandas as pd
import statsmodels.api as sm

# Part 1

data = pd.read_excel(r'C:\Users\kyungmyung\Downloads\Financial_Data.xlsx')
data = data.dropna(subset=['Exchange Rate Effect','Accrued Expenses','Selling, General and Administrative Expense','Sales/Turnover (Net)'])

data.reset_index(drop=True)

exog_vars = ['Exchange Rate Effect','Accrued Expenses','Selling, General and Administrative Expense']
exog = sm.add_constant(data[exog_vars])

model = sm.OLS(data['Sales/Turnover (Net)'], exog)
pooled_regression = model.fit()
pooled_regression.summary()

# Part 2

audit_data = pd.read_csv(r'C:\Users\kyungmyung\Downloads\Audit_CSV.csv')

audit_exog_vars = ['IT_FEES','TAX_FEES', 'RESTATEMENT']
audit_exog = sm.add_constant(audit_data[audit_exog_vars])

audit_model = sm.OLS(audit_data['AUDIT_FEES'], audit_exog)
audit_pooled_regression = audit_model.fit()
audit_pooled_regression.summary()
