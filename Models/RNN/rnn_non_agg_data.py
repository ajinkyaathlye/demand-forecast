import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

table_df = pd.read_csv('.../Data/table_data_rnn.csv', index_col='Date')
# Replacing erroneous 172 CPI values with correct value 2
table_df.loc[table_df['CPI'] == 172, 'CPI'] = 2
# Replacing erroneous 16455.38 GDP value for 2010-01 with 0
table_df.loc[table_df['GDP'] == 16455.38, 'GDP'] = 0

table_df.drop(['Index', 'GDP'], inplace=True, axis=1)
table_df.index = pd.to_datetime(table_df.index)
table_df.sort_index(inplace=True)
#
table_df['CPI'] = 2*((table_df['CPI'] - table_df['CPI'].min())/(table_df['CPI'].max() - table_df['CPI'].min())) - 1

# print table_df
# Values to be untouched, test set is used for plotting.
train_set = table_df.head(84)
test_set = table_df.tail(12)

model_train_set = table_df.head(84)
model_test_set = table_df.tail(12)
# print train_set
scaler = MinMaxScaler()
train_temp_scaled = scaler.fit_transform(model_train_set[['Quantity', 'Cost']])
# print train_temp_scaled
test_temp_scaled = scaler.transform(model_test_set[['Quantity', 'Cost']])
model_train_set[['Quantity', 'Cost']] = train_temp_scaled
model_test_set[['Quantity', 'Cost']] = test_temp_scaled

train_scaled = np.array(model_train_set)
test_scaled = np.array(model_test_set)
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
# print table_df
train_set = table_df.head(84)
test_set = table_df.tail(12)
#
# # print train_set
#
# scaler = MinMaxScaler()
# train_scaled = scaler.fit_transform(train_set)
# test_scaled = scaler.transform(test_set)
# np.set_printoptions(formatter={'float_kind':'{:f}'.format})
#

# def next_batch(training_data, steps):
#     # Grab a random starting point for each batch
#     rand_start = np.random.randint(0, len(training_data) - steps)
#     # rand_start = 0
#     # Create Y data for time series in the batches
#     y_batch = np.array(training_data[rand_start:rand_start + steps + 1]).reshape(-1, 4)
#     # print y_batch.shape
#     return y_batch[:-1, :].reshape(-1, steps, 4), y_batch[1:, 2].reshape(-1, steps, 1)
#
#
# # print next_batch(train_set, 12)
#
# # Just one feature, the time series
# num_inputs = 4
# # Num of steps in each batch
# num_time_steps = 12
# # 100 neuron layer, play with this
# num_neurons = 100
# # Just one output, predicted time series
# num_outputs = 1
#
# ## You can also try increasing iterations, but decreasing learning rate
# # learning rate you can play with this
# learning_rate = 0.01
# # how many iterations to go through (training steps), you can play with this
# num_train_iterations = 4000
# # Size of the batch of data
# batch_size = 5
# tf.reset_default_graph()
# X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
# y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])
#
# cell = tf.contrib.rnn.OutputProjectionWrapper(
#     tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu),
#     output_size=num_outputs)
#
# outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
#
# loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     for iteration in range(num_train_iterations):
#
#         X_batch, y_batch = next_batch(train_scaled, num_time_steps)
#         sess.run(train, feed_dict={X: X_batch, y: y_batch})
#
#         if iteration % 100 == 0:
#             mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
#             print(iteration, "\tMSE:", mse)
#
#     # Save Model for Later
#     saver.save(sess, "./ex_time_series_model")
#
# with tf.Session() as sess:
#     # Use your Saver instance to restore your saved rnn time series model
#     saver.restore(sess, "./ex_time_series_model")
#
#     # Create a numpy array for your generative seed from the last 12 months of the
#     # training set data. Hint: Just use tail(12) and then pass it to an np.array
#     train_seed = list(train_scaled[-12:])
#     test_seed = list(test_scaled)
#     # Now create a for loop that
#     for iteration in range(12):
#         # print "------------------"
#         # print type(np.array(train_seed[-num_time_steps:])[0])
#
#         X_batch = np.array(np.array([train_seed[-num_time_steps:]]))
#         y_pred = sess.run(outputs, feed_dict={X: X_batch})
#         # print y_pred
#         new_X = test_seed[iteration]
#         new_X[2] = y_pred[0, -1, 0]
#         train_seed.append(new_X)
#         # print train_seed
#
# plot_seed = np.array(train_seed)
# # print train_seed[12:,2].reshape(12,1)
# # print np.array(np.array(train_seed[12:]))
# results = scaler.inverse_transform(train_seed[12:])
# test_set['Generated'] = results[:, 2].reshape(12,1)
# test_set.drop(['Cost', 'GDP', 'CPI'], inplace=True, axis=1)
# test_set.plot()
# plt.show()