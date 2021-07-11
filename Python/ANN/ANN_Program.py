from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_excel(
    ("E:\Kuliah\Github\School\School\Python\ANN\InputSet3.xlsx"), engine='openpyxl',)

# splitting the remaining data in training data and validation data.
df_train, df_test = train_test_split(df, test_size=0.3, shuffle=False)

df_train.index = range(df_train.shape[0])
df_test.index = range(df_test.shape[0])

# Scaling the data using MinMax Scaler.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_train.drop(["Y"], axis=1).values)
Y_train = scaler.fit_transform(df_train["Y"].values.reshape(-1, 1))
X_test = scaler.fit_transform(df_test.drop(["Y"], axis=1).values)
Y_test = scaler.fit_transform(df_test["Y"].values.reshape(-1, 1))

# function to denormalise the predicted values.


def denormalize(df, norm_data):
    df = df["Y"].values.reshape(-1, 1)
    norm_data = norm_data.reshape(-1, 1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)

    return new

  # Reshaping the data into [samples, test_size, n_features] suitable for LSTM model.
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# defining hyper parameters for LSTM model.
epochs = 50
n_hidden = 50
n_features = 4
batch_size = 100
train_loss = []

# to reset tensorflow graph for every run.
tf.reset_default_graph()


# variables required for model creation.
x_data = tf.placeholder('float', [None, n_features, 1])
y_target = tf.placeholder('float')

W = tf.Variable(tf.random_normal([n_hidden, 1]))
bias = tf.Variable(tf.random_normal([1]))

# 1-layer LSTM with n_hidden units.
rnn_cell = rnn.BasicLSTMCell(n_hidden)


def rnn_model(x):

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.unstack(x, n_features, 1)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_features outputs but
    # we only want the last output
    return (tf.matmul(outputs[-1], W) + bias)


# output equation of LSTM model.
y_predicted = tf.reshape(rnn_model(x_data), [-1])

# loss function = MSE (Mean Squared Error)
cost = tf.reduce_mean(tf.square(y_predicted-y_target))

# Using Adam as the optimization algorithm.
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
# tf.train.GradientDescentOptimizer(0.01).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for epoch in range(epochs):

    start = random.randint(0, (len(X_train)-batch_size))
    end = start + batch_size

    batch_x = np.array(X_train[start:end])
    batch_y = np.array(Y_train[start:end])

    # loop for training each batch.
    for j in range(batch_x.shape[0]):
        sess.run(optimizer, feed_dict={
                 x_data: batch_x[j].reshape(1, 4, 1), y_target: batch_y[j]})
    # loop for training each batch ends. 

    train_loss.append(
        sess.run(cost, feed_dict={x_data: X_train, y_target: Y_train}))

    print('Epoch', epoch, 'completed out of', epochs,
          'Training_loss:', train_loss[epoch])

# stores the predicted value for test data.
pred = sess.run(y_predicted, feed_dict={x_data: X_test})

# denormalizing our predicted value.
y_test = denormalize(df_test, Y_test)
pred = denormalize(df_test, pred)

# plot showing difference between actual test data and predicted test data.
plt.figure(figsize=[5*1.5, 3*1.5])
plt.plot(range(y_test.shape[0]), y_test, label="Original Data")
plt.plot(range(y_test.shape[0]), pred, label="Predicted Data")
plt.legend(loc='best')
plt.ylabel('Amount of Rainfall (in mm)', fontsize="14")
plt.xlabel('Time (Days)', fontsize="14")
plt.show()

print("MAE : ", np.mean(abs(y_test-pred)))

print("RMSE : ", np.sqrt(np.mean(np.square(y_test-pred))))
