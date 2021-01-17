import pandas as pd
import numpy as np
from pandas import datetime
import warnings
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


series = pd.read_csv('../../Data/Kaggle Dataset/merged_data_set.csv', usecols=['Weekly_Sales', 'Date', 'Store'], index_col='Date')
series.sort_index(inplace=True)
series = series.loc[series['Store'] == 1]
series = series.groupby(['Date']).sum().astype('float64')
series.drop('Store', axis=1, inplace=True)
series['Weekly_Sales'] = series['Weekly_Sales'] / 10000.
series = series.loc[series['Weekly_Sales'] >= 1]

# print series
# evaluate parameters
# p_values = [1, 6, 29, 46, 52]
# d_values = range(0, 2)
# q_values = range(0, 2)
# warnings.filterwarnings("ignore")
# evaluate_models(series.values, p_values, d_values, q_values)

model = ARIMA(series, order=(1, 1, 1))
model_fit = model.fit(disp=0)
print "\n\n\n\n\n\n"
print(model_fit.summary())

# plot residual errors
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show()
# residuals.plot(kind='kde')
# pyplot.show()
# autocorrelation_plot(series)
# pyplot.show()

# print(residuals.describe())

print series
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(1, 1, 1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0] if output[0] > 0 else 0
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)

pred_df = pd.DataFrame(predictions).tail(12)
test_df = pd.DataFrame(test).tail(12)
test_df.rename(index=str, columns={0: "Weekly Sales"}, inplace=True)
pred_df['Date'] = series.tail(12).index
test_df['Date'] = series.tail(12).index
test_df['Date'] = pd.to_datetime(test_df['Date'])
pred_df['Date'] = pd.to_datetime(pred_df['Date'])
pred_df.set_index('Date', inplace=True)
test_df.set_index('Date', inplace=True)
test_df['Generated'] = pred_df



print test_df

accuracy = (pred_df[0] - test_df['Weekly Sales']).abs() / test_df['Weekly Sales']
print "Accuracy: "
print (accuracy.mean()) * 100


# plot
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')

test_df.plot()
pyplot.show()