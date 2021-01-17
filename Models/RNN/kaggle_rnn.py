import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.read_csv('../../Data/Kaggle Dataset/merged_data_set.csv')

df.drop_duplicates(inplace=True)
# df.index = pd.to_datetime(df.index)
# df.sort_index(inplace=True)
df.drop('Unnamed: 0', axis=1, inplace=True)
# ['Date', 'IsHoliday', 'CPI', 'Temperature', 'Unemployment', 'Store']
# df.reset_index(level=['IsHoliday', 'CPI', 'Temperature', 'Unemployment', 'Store'], inplace=True)
# print df
df['IsHoliday'] = df['IsHoliday'].apply(lambda x: 1 if x == 'True' else 0)
# print df
table_df = df.loc[df['Store'] == 1]
table_df.drop('Store', axis=1, inplace=True)
table_df = table_df.groupby(by=['Date']).sum()
# table_df.reset_index(level=['Temperature', 'CPI', 'Unemployment', 'IsHoliday'], inplace=True)
table_df['Weekly_Sales'] = table_df['Weekly_Sales'] / 10000.
table_df = table_df.loc[table_df['Weekly_Sales'] >= 1]
plt.figure(figsize=(20, 5))
plt.grid()
table_df.index = pd.to_datetime(table_df.index)
table_df['Weekly_Sales'].plot()
# plt.show()
# print table_df
train_percent = int(len(table_df) - 12)
test_percent = 12
train_set = table_df.head(train_percent)
test_set = table_df.tail(test_percent)
print(train_set)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
# print train_temp_scaled
test_scaled = scaler.transform(test_set)
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

def next_batch(training_data, steps):
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0, len(training_data) - steps)
    # Create Y data for time series in the batches
    y_batch = np.array(training_data[rand_start:rand_start + steps + 1]).reshape(-1, 5)
    # print y_batch[:-6, :].reshape(-1, steps, 5)
    return y_batch[:-1, :].reshape(-1, steps, 5), y_batch[1:, 0].reshape(-1, steps, 1)

# def next_batch(training_data, steps):
#     # Grab a random starting point for each batch
#     rand_start = np.random.randint(0, len(training_data) - steps - 6)
#     # Create Y data for time series in the batches
#     y_batch = np.array(training_data[rand_start:rand_start + steps + 6]).reshape(-1, 5)
#     return y_batch[:-6, :].reshape(-1, steps, 5), y_batch[6:, 0].reshape(-1, steps, 1)

# next_batch(train_set, 12)
# Just one feature, the time series
num_inputs = 5
# Num of steps in each batch
num_time_steps = 12
# 100 neuron layer, play with this
num_neurons = 100
# Just one output, predicted time series
num_outputs = 1

# You can also try increasing iterations, but decreasing learning rate
# learning rate you can play with this
learning_rate = 0.01
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 4000
# Size of the batch of data
batch_size = 5
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.tanh),
    output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# with tf.Session() as sess:
#     sess.run(init)
#     for iteration in range(num_train_iterations):
#
#         X_batch, y_batch = next_batch(train_scaled, num_time_steps)
#         sess.run(train, feed_dict={X: X_batch, y: y_batch})
#
#         if iteration % 100 == 0:
#             mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
#             print(iteration, "\tMSE:", mse)
#
#     saver.save(sess, "./kaggle_rnn")

with tf.Session() as sess:
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./kaggle_rnn")

    # Create a numpy array for your generative seed from the last 12 months of the
    # training set data. Hint: Just use tail(12) and then pass it to an np.array
    train_seed = list(train_scaled[-12:])
    # print train_seed
    test_seed = list(test_scaled)

    # Now create a for loop that
    for iteration in range(12):
        # Input needs to be a 3-D array.
        X_batch = np.array(np.array([train_seed[-num_time_steps:]]))
        # print X_batch
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        # Creating a new array and feeds the next value of input data from test_seed and y_pred (quantity)
        # for getting all inputs one time-step in the future to make a prediction.
        new_X = test_seed[iteration]
        new_X[0] = y_pred[0, -1, 0]
        train_seed.append(new_X)
        # print train_seed


# print train_seed[12:]
results = scaler.inverse_transform(train_seed[12:])
# print results[:, 0].reshape(12, 1)
# print test_set
test_set['Generated'] = results[:, 0].reshape(12, 1)
# print test_set
# plt.figure(figsize=(20, 5))
# test_set.drop(['Cost', 'GDP', 'CPI'], inplace=True, axis=1)
test_set.index = pd.to_datetime(test_set.index)
test_set[['Generated', 'Weekly_Sales']].plot()
accuracy = (test_set['Generated'] - test_set['Weekly_Sales']).abs() / test_set['Weekly_Sales']
print(accuracy.mean() * 100)
plt.show()