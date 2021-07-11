from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import tkinter as tk
import matplotlib.pyplot as pl
from tkinter import filedialog
from numpy import zeros, concatenate, ravel, diff, array, ones
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

# Windows
Window = tk.Tk()

# Judul & Ukuran
Window.title("ANN Rainfall Precipitation")
Window.geometry('800x600')

# Frame1
Frame1 = tk.Frame(Window)
Frame1.pack()

def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Text files",
                                                        "*.xlsx*"),
                                                       ("all files",
                                                        "*.*")))
    return filename

open = tk.Button(Frame1, text="Open File", command=browseFiles)
open.grid(row=0, column=0, sticky="w")

df = pd.read_excel(
    (browseFiles()), engine='openpyxl',)


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



Frame2 = tk.Frame(Window)
Frame2.pack(side='top', anchor='w')

Frame3 = tk.Frame(Window)
Frame3.pack(side='top', anchor='center')

def Training():
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
        
        print('Epoch', epoch, 'completed out of', epochs, 'Training_loss:', train_loss[epoch])
        # fig = Figure(figsize = (5*1.5, 3*1.5), 
        #          dpi = 100) 

        # pl2 = fig.add_subplot(111) 

        # pl2.plot(X_test, label="Test")
        # pl2.plot(X_train, label="Train")
        # pl2.legend(loc='best')
        # pl2.set_xlabel('Training Loss', fontsize="14")

        # canvas1 = FigureCanvasTkAgg(fig, master = Frame3)   
        # canvas1.get_tk_widget().grid(row =2,column=0,sticky="w")
 
        # train1 = tk.Label(Frame2, text="Training Completed", font="Arial 10 bold")
        # train1.grid(row=1, column=0, sticky="w")

def Graph():
    # stores the predicted value for test data.
    pred = sess.run(y_predicted, feed_dict={x_data: X_test})

    # denormalizing our predicted value.
    y_test = denormalize(df_test, Y_test)
    pred = denormalize(df_test, pred)

    mae = np.mean(abs(y_test-pred))
    rms =  np.sqrt(np.mean(np.square(y_test-pred)))
    maed = tk.StringVar(value=mae)
    rmsed = tk.StringVar(value=rms)
    
    label1 = tk.Label(Frame2, text="MAE: ", font="Arial 10 bold")
    label1.grid(row=2, column=0, sticky="w")

    label2 = tk.Label(Frame2, textvariable=maed, font="Arial 10 bold")
    label2.grid(row=2, column=1, sticky="w")

    label3 = tk.Label(Frame2, text="RMSE: ", font="Arial 10 bold")
    label3.grid(row=3, column=0, sticky="w")

    label4 = tk.Label(Frame2, textvariable=rmsed, font="Arial 10 bold")
    label4.grid(row=3, column=1, sticky="w")

    label3 = tk.Label(Frame2, text="mm", font="Arial 10 bold")
    label3.grid(row=3, column=2, sticky="w")

    fig = Figure(figsize = (5*1.5, 3*1.5), 
                 dpi = 100) 

    pl1 = fig.add_subplot(111) 

    pl1.plot(range(y_test.shape[0]), y_test, label="Original Data")
    pl1.plot(range(y_test.shape[0]), pred, label="Predicted Data")
    pl1.legend(loc='best')
    pl1.set_ylabel('Amount of Rainfall (in mm)', fontsize="14")
    pl1.set_xlabel('Time (Days)', fontsize="14")

    canvas1 = FigureCanvasTkAgg(fig, master = Frame3)   
    canvas1.get_tk_widget().grid(row =1,column=0,sticky="w")


Button1 = tk.Button(Frame1, text="Training", command=Training)
Button1.grid(row=0, column=1, sticky="w")

Button2 = tk.Button(Frame1, text="Graph", command=Graph)
Button2.grid(row=0, column=3, sticky="w")

Window.mainloop()
