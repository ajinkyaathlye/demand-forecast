# RNN model for forecasting Quantity for next 12 months using
# GDP, CPI, Quantity, Cost as parameters. predicting 1 time-step in the future


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

table_df = pd.read_csv('../../Data/table_data_rnn.csv', index_col='Date')
# Replacing erroneous 172 CPI values with correct value 2
table_df.loc[(table_df['CPI'] == 172) | (table_df['CPI'] == 286), 'CPI'] = 2
# Replacing erroneous 16455.38 GDP value for 2010-01 with 0
table_df.loc[table_df['GDP'] == 16455.38, 'GDP'] = 0

table_df.drop('Index', inplace=True, axis=1)
table_df.index = pd.to_datetime(table_df.index)
table_df = table_df.groupby(['Date', 'GDP', 'CPI']).sum()
table_df.reset_index(level=['GDP', 'CPI'], inplace=True)

# print table_df
# Values to be untouched, test set is used for plotting.
train_set = table_df.head(84)
test_set = table_df.tail(12)

# print train_set
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
# print train_temp_scaled
test_scaled = scaler.transform(test_set)
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

def next_batch(training_data, steps):
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0, len(training_data) - steps)
    # rand_start = 0
    # Create Y data for time series in the batches
    y_batch = np.array(training_data[rand_start:rand_start + steps + 1]).reshape(-1, 4)
    # print y_batch.shape
    return y_batch[:-1, :].reshape(-1, steps, 4), y_batch[1:, 2].reshape(-1, steps, 1)



# print next_batch(train_set, 12)

# Just one feature, the time series
num_inputs = 4
# Num of steps in each batch
num_time_steps = 12
# 100 neuron layer, play with this
num_neurons = 20
# Just one output, predicted time series
num_outputs = 1

## You can also try increasing iterations, but decreasing learning rate
# learning rate you can play with this
learning_rate = 0.005
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 6000
# Size of the batch of data
batch_size = 5
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_train_iterations):

        X_batch, y_batch = next_batch(train_scaled, num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})

        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    saver.save(sess, "./rnn_with_gdp")

with tf.Session() as sess:
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./rnn_with_gdp")

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
        new_X[2] = y_pred[0, -1, 0]
        train_seed.append(new_X)
        # print train_seed

print train_seed[12:]
results = scaler.inverse_transform(train_seed[12:])
print results
test_set['Generated'] = results[:, 2].reshape(12,1)
test_set.drop(['Cost', 'GDP', 'CPI'], inplace=True, axis=1)
test_set.plot()
plt.show()