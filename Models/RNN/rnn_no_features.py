import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.read_csv('../../Data/table_data_rnn.csv', index_col='Date')
df.index = pd.to_datetime(df.index)
df.drop(['Index', 'GDP', 'Cost', 'CPI'], inplace=True, axis=1)
train_set = df.head(84)
test_set = df.tail(12)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)


def next_batch(training_data, batch_size, steps):
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0, len(training_data) - steps)

    # Create Y data for time series in the batches
    y_batch = np.array(training_data[rand_start:rand_start + steps + 1]).reshape(1, steps + 1)

    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)



# Just one feature, the time series
num_inputs = 1
# Num of steps in each batch
num_time_steps = 12
# 100 neuron layer, play with this
num_neurons = 100
# Just one output, predicted time series
num_outputs = 1

## You can also try increasing iterations, but decreasing learning rate
# learning rate you can play with this
learning_rate = 0.001
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 6000
# Size of the batch of data
batch_size = 5

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
#
with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_train_iterations):

        X_batch, y_batch = next_batch(train_scaled, batch_size, num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})

        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    # Save Model for Later
    saver.save(sess, "./rnn_no_features")

with tf.Session() as sess:
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./rnn_no_features")

    # Create a numpy array for your generative seed from the last 12 months of the
    # training set data. Hint: Just use tail(12) and then pass it to an np.array
    train_seed = list(train_scaled[-12:])
    test_seed = list(test_scaled)

    ## Now create a for loop that
    for iteration in range(12):
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)

        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        print X_batch
        print y_pred
        print "------------------------------"
        train_seed.append(y_pred[0, -1, 0])

results = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))
print results
test_set['Generated'] = results
test_set.plot()
plt.show()
