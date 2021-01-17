import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import iterator_class
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.python.util import compat
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from scipy import stats



tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS


# Import the data
df = pd.read_csv('../Data/Kaggle Dataset/merged_data_set.csv')

df.drop_duplicates(inplace=True)
df.drop('Unnamed: 0', axis=1, inplace=True)
df['IsHoliday'] = df['IsHoliday'].apply(lambda x: 1 if x == 'True' else 0)
table_df = df.loc[df['Store'] == 1]
table_df.drop('Store', axis=1, inplace=True)
table_df = table_df.groupby(by=['Date', 'CPI', 'Unemployment', 'Temperature']).sum()
table_df.reset_index(level=['Temperature', 'CPI', 'Unemployment'], inplace=True)
table_df['Weekly_Sales'] = table_df['Weekly_Sales'] / 10000.
table_df = table_df.loc[table_df['Weekly_Sales'] >= 1]
shifted_sales = table_df['Weekly_Sales'].shift(1)
table_df['Weekly_Sales'] = shifted_sales
table_df.dropna(inplace=True)
# print table_df
# raw_csv_data = np.loadtxt("../Data/categorised_data.csv", delimiter=',')
raw_csv_data = np.array(table_df)
raw_target_data = raw_csv_data[:, 3]
raw_train_data = np.delete(raw_csv_data, 3, 1)
# print raw_train_data
# print raw_target_data
# Divide the cost column by quantity to get unit cost of item
# raw_train_data[:, 0] = raw_train_data[:, 0] / raw_target_data


# Shuffle the dataset prior to balancing as we want to utilise as much data as possible for 0's
shuffled_indices_for_raw_data = np.arange(raw_target_data.shape[0])
np.random.shuffle(shuffled_indices_for_raw_data)
raw_train_data = raw_train_data[shuffled_indices_for_raw_data]
raw_target_data = raw_target_data[shuffled_indices_for_raw_data]

# Split the dataset into training, testing, and validation

samples_count = raw_train_data.shape[0]

train_count = int(0.9*samples_count)
test_count = int(0.1*samples_count)
# validation_count = samples_count - train_count - test_count

input_train_data = raw_train_data[:train_count]
target_train_data = raw_target_data[:train_count]

input_test_data = raw_train_data[train_count:train_count + test_count]
target_test_data = raw_target_data[train_count:train_count + test_count]

input_scaler = MinMaxScaler()
input_train_data = input_scaler.fit_transform(input_train_data)
target_train_data = input_scaler.fit_transform(target_train_data.reshape(-1, 1))
input_test_data = input_scaler.transform(input_test_data)
target_test_data = input_scaler.transform(target_test_data.reshape(-1, 1))

# print target_train_data

# print scaled_training_data.shape
# indices_array = np.arange(scaled_training_data.shape[0])
# np.random.shuffle(indices_array)
# Rearranges the array directly according to the array passed
# shuffled_inputs = scaled_training_data[indices_array]
# shuffled_targets = raw_target_data[indices_array]

# input_validation_data = shuffled_inputs[train_count + test_count:]
# target_validation_data = shuffled_targets[train_count + test_count:]

np.savez("../Data/NPZ_data/Sanskriti_training_data", inputs = input_train_data, targets = target_train_data)
np.savez("../Data/NPZ_data/Sanskriti_test_data", inputs = input_test_data, targets = target_test_data)
# np.savez("../Data/NPZ_data/Sanskriti_validation_data", inputs = input_validation_data, targets = target_validation_data)

# ----------------------END OF PRE-PROCESSING---------------------------

# -----------------------------------
# Build model
# -----------------------------------
def main(_):

    input_size = 4
    hiddenlayer_size = 10
    output_size = 1
    batch_size = 12
    max_epochs = 500
    number_hidden_layer = 5

    tf.reset_default_graph()

    inputs = tf.placeholder(tf.float32, [None, input_size])
    targets = tf.placeholder(tf.float32, None)

    start_weights = tf.get_variable("start_weights", [input_size, hiddenlayer_size])
    start_bias = tf.get_variable("start_bias", [hiddenlayer_size])

    # print inputs, weights0
    Input_next = tf.nn.relu(tf.matmul(inputs, start_weights) + start_bias)

    for i in range(number_hidden_layer - 1): #-1 since our first hidden layer is created outside the for loop.
        weights = 'weights{}'.format(str(i))
        bias = 'bias{}'.format(str(i))
        Weights = tf.get_variable(weights, [hiddenlayer_size, hiddenlayer_size])
        Bias = tf.get_variable(bias, [hiddenlayer_size])
        Input_next = tf.nn.relu(tf.matmul(Input_next, Weights) + Bias)

    final_weight = tf.get_variable("final_weight", [hiddenlayer_size, output_size])
    final_bias = tf.get_variable("final_bias", [output_size])
    # print hidden_layer
    # Final layer matmul of weights and biases which is later used to calculate the activation values.
    outputs = tf.matmul(Input_next, final_weight) + final_bias

    loss = tf.reduce_mean(tf.square(tf.subtract(outputs, targets)))

    optimizer = tf.train.AdamOptimizer(.001).minimize(loss)
    equal_vector = tf.divide(tf.abs(tf.subtract(outputs, targets)), targets)
    accuracy = tf.reduce_mean(tf.cast(equal_vector, tf.float64))

    sess = tf.InteractiveSession()
    initializer = tf.global_variables_initializer()

    sess.run(initializer)

    # Setup interactive plot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(target_test_data)
    line2, = ax1.plot(target_test_data * 0.5)
    plt.show()

    # prev_validation_loss = 999999999.

    # Loads data using the Audiobooks_Data_Reader class.
    train_data = iterator_class.Sanskriti_Data_Reader('training', batch_size=batch_size)
    # print train_data.targets
    # validation_data = iterator_class.Sanskriti_Data_Reader('validation')
    test_data = iterator_class.Sanskriti_Data_Reader('test')

    #Training the model.
    for epoch in range(max_epochs):
        current_epoch_loss = 0

        for i, input_batch, target_batch in train_data:
            # print input_batch.shape
            _, batch_loss, output_batch = sess.run([optimizer, loss, outputs], feed_dict = {
                inputs: input_batch, targets: target_batch
            })
            # print target_batch
            current_epoch_loss += batch_loss
            # if epoch % 50 == 0:
            #     # Prediction
            #     pred = sess.run(outputs, feed_dict={inputs: input_test_data})
            #     line2.set_ydata(pred)
            #
            #     plt.title('Epoch ' + str(epoch))
            #
            #     # file_name = '../Data/Data_Visualization/Images/epoch_' + str(epoch) + '_batch_' + str(i) + '.png'
            #     # plt.savefig(file_name)
            #     # plt.pause(0.01)
            #     df = pd.DataFrame(pred)
            #     test_df = pd.DataFrame(input_test_data)
            #     df[0].plot()
            #     test_df[0].plot()
            #     plt.show()

            if epoch % 100 == 0:
                print "Actual value: {}\n Predicted value {}\n".format(
                    input_scaler.inverse_transform(target_batch),
                    input_scaler.inverse_transform(output_batch)
                )
                print "=========================================================="


        current_epoch_loss /= train_data.batch_count # train_data is the class instance of Sanskriti Data Reader class.


        # Validating the model with validation data.
        # validation_loss = 0.
        #
        # for input_batch, target_batch in validation_data:
        #     validation_loss = sess.run([loss], feed_dict = {
        #         inputs: input_batch, targets: target_batch
        #     })

        print "Epoch: {}".format(str(epoch + 1))
        print("Epoch loss for training set: {}".format(
            current_epoch_loss,
        ))
        # for i in xrange(15):
        #     print "Actual value: {}\n Predicted value {}\n".format(target_batch[i], output_batch[i])
        # print "=============================================================="

        # if prev_validation_loss < validation_loss:
        #     break
        #
        # prev_validation_loss = validation_loss
    for i, input_batch, target_batch in test_data:
        test_accuracy, eq_vec = sess.run([accuracy, equal_vector], feed_dict={
            inputs: input_batch, targets: target_batch
        })
        print eq_vec

        print "Test accuracy: {}%".format(test_accuracy * 100)
    # # -----------------------------------
    # Save model: Used tensorflow serving to build a servable.
    # -----------------------------------

    # export_path_base = '/home/hp/PycharmProjects/Demand Forecast/Models'
    # export_path = os.path.join(
    #     compat.as_bytes(export_path_base),
    #     compat.as_bytes(str(FLAGS.model_version)))
    # print 'Exporting trained model to', export_path
    #
    # builder = saved_model_builder.SavedModelBuilder(export_path)
    # # NOTE: What are the inputs/target/outputs to be used? Placeholder? or actual value?
    # # NOTE: Convert numpy array to tensor and then create a signature map
    # # Use 'PREDICT_INPUT / PREDICT_OUTPUT' since ours is a regression problem.
    # classification_inputs = utils.build_tensor_info(tf.convert_to_tensor(inputs))
    # classification_outputs_scores = utils.build_tensor_info(tf.convert_to_tensor(outputs))
    #
    # prediction_signature = signature_def_utils.build_signature_def(
    #     inputs={'items': classification_inputs},
    #     outputs={'predicted_value': classification_outputs_scores},
    #     method_name=signature_constants.PREDICT_METHOD_NAME)
    #
    # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    #
    # # add the sigs to the servable
    # builder.add_meta_graph_and_variables(
    #     sess, [tag_constants.SERVING],
    #     signature_def_map={
    #         signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #             prediction_signature,
    #     },
    #     legacy_init_op=legacy_init_op)
    #
    # # save it!
    # builder.save()


if __name__ == '__main__':
  tf.app.run()