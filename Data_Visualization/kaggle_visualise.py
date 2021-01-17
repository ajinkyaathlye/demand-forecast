import numpy as np
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
"""
sales = pd.read_csv('../Data/Kaggle Dataset/sales data-set.csv')
features = pd.read_csv('../Data/Kaggle Dataset/Features data set.csv')
# sales = sales.loc[(sales['Store'] == 1) | (sales['Store'] == 2)]
# features = features.loc[(features['Store'] == 1) & (features['Store'] == 2)]
sales.drop('Dept', inplace=True, axis=1)
features.drop(['Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], inplace=True, axis=1)
sales['Date'] = pd.to_datetime(sales['Date'])
features['Date'] = pd.to_datetime(features['Date'])
# df.sort_index(inplace=True)
df = pd.merge(sales, features, on=['Store', 'Date', 'IsHoliday'], how='left')
df = df.fillna(0)
df['Temperature'] = (df['Temperature'] - 32) * 5./9.

df = df.loc[(df['Store'] == 1) | (df['Store'] == 2)]
df.to_csv('../Data/Kaggle Dataset/merged_data_set.csv')"""

# df[['Date', 'Temperature', 'CPI', 'Unemployment']].plot(x='Date', subplots=True, figsize=(20, 15))
# plt.show()


def fit_ar_model(ts, orders):
    X = np.array(
        [ts.values[(i - orders)].squeeze() if i >= np.max(orders) else np.array(len(orders) * [np.nan]) for i in
         range(len(ts))])

    mask = ~np.isnan(X[:, :1]).squeeze()

    Y = ts.values

    lin_reg = LinearRegression()

    lin_reg.fit(X[mask], Y[mask])

    print(lin_reg.coef_, lin_reg.intercept_)

    print('Score factor: %.2f' % lin_reg.score(X[mask], Y[mask]))

    return lin_reg.coef_, lin_reg.intercept_


def predict_ar_model(ts, orders, coef, intercept):
    return np.array(
        [np.sum(np.dot(coef, ts.values[(i - orders)].squeeze())) + intercept if i >= np.max(orders) else np.nan for i in
         range(len(ts))])


df = pd.read_csv('../Data/Kaggle Dataset/merged_data_set.csv')
df.drop_duplicates(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df_average_sales_week = df.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
df_average_sales = df_average_sales_week.sort_values('Weekly_Sales', ascending=False)


# plt.figure(figsize=(20, 5))
# plt.plot(df_average_sales_week.Date, df_average_sales_week.Weekly_Sales)
# plt.show()
#
# print df_average_sales

ts = df_average_sales_week.set_index('Date')

#



df1 = df.where(df['Store'] == 1)
df1 = df1.dropna()
df1 = df1.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
df1 = df1.set_index('Date')
df1['Weekly_Sales'] = df1['Weekly_Sales'] / 10000.
df1 = df1.loc[df1['Weekly_Sales'] >= 1]

fig, axes = plt.subplots(1, 2, figsize=(20, 5))
plot_acf(df1, lags=100, ax=axes[0])
plot_pacf(df1, lags=100, ax=axes[1])
plt.show()

orders = np.array([1, 6, 29, 46, 52])
coef, intercept = fit_ar_model(df1, orders)
pred = pd.DataFrame(index=df1.index, data=predict_ar_model(df1, orders, coef, intercept))

# diff = (df1['Weekly_Sales'] - pred[0]) / df1['Weekly_Sales']

# print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std()))

"""
plt.figure(figsize=(20, 5))
plt.plot(diff, c='orange')
plt.grid()
# df.loc[df['Weekly_Sales'] == 1].plot()
plt.show()
"""


def fit_ar_model_ext(ts, orders, ext, fitter=LinearRegression()):
    X = np.array(
        [ts.values[(i - orders)].squeeze() if i >= np.max(orders) else np.array(len(orders) * [np.nan]) for i in
         range(len(ts))])

    X = np.append(X, ext.values, axis=1)

    mask = ~np.isnan(X[:, :1]).squeeze()

    Y = ts.values

    fitter.fit(X[mask], Y[mask].ravel())

    print(fitter.coef_, fitter.intercept_)

    print('Score factor: %.2f' % fitter.score(X[mask], Y[mask]))

    return fitter.coef_, fitter.intercept_


def predict_ar_model_ext(ts, orders, ext, coef, intercept):
    X = np.array(
        [ts.values[(i - orders)].squeeze() if i >= np.max(orders) else np.array(len(orders) * [np.nan]) for i in
         range(len(ts))])

    X = np.append(X, ext.values, axis=1)

    return np.array(np.dot(X, coef.T) + intercept)


dfext = df.where(df['Store'] == 1)
dfext = dfext.dropna()
dfext = dfext.groupby(by=['Date'], as_index=False)[['Temperature', 'CPI', 'Unemployment']].mean()
dfext = dfext.set_index('Date')
# print dfext
dfext['shifted_sales'] = df1.shift(-1)

# Regression analysis
# corr = dfext.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(corr,
#             annot=True, fmt=".3f",
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
# plt.show()
#
# corr['shifted_sales'].sort_values(ascending=False)

dfexte = dfext[['Unemployment', 'CPI', 'Temperature']]

orders = np.array([1, 6, 29, 46, 52])
coef, intercept = fit_ar_model_ext(df1,orders,dfexte)
pred_ext=pd.DataFrame(index=df1.index, data=predict_ar_model_ext(df1, orders, dfexte, coef, intercept))
plt.figure(figsize=(20,5))
# plt.plot(df1.tail(12))
# plt.plot(pred.tail(12))
# plt.plot(pred_ext.tail(12))
df_plot = pd.DataFrame()
# print df1.tail(12)
df_plot['Multivariate'] = pred_ext[0].tail(12)
df_plot['Univariate'] = pred[0].tail(12)
df_plot['Weekly Sales'] = df1['Weekly_Sales'].tail(12)

df_plot['Date'] = df1.tail(12).index
# df_plot['Date'] = pd.to_datetime(df['Date'])
df_plot.set_index('Date', inplace=True)
print df_plot
df_plot.plot()

accuracy = (pred_ext.tail(12)[0] - df1.tail(12)['Weekly_Sales']).abs() / pred_ext.tail(12)[0]
# print df1.tail(12)['Weekly_Sales']
print accuracy
print "Accuracy: "
print (accuracy.mean()) * 100

plt.show()

diff = (df1['Weekly_Sales'] - pred[0]) / df1['Weekly_Sales']
diff_ext = (df1['Weekly_Sales'] - pred_ext[0]) / df1['Weekly_Sales']

print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std()))
print('AR with Ext Residuals: avg %.2f, std %.2f' % (diff_ext.mean(), diff_ext.std()))

# plt.figure(figsize=(20, 5))
# plt.plot(diff, c='orange', label='w/o external variables')
# plt.plot(diff_ext, c='green', label='w/ external variables')
# # plt.plot(plot_ext)
# plt.legend()
# plt.grid()
# plt.show()