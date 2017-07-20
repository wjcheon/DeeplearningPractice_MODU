#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 07:36:47 2017

@author: Wonjoong Cheon
"""
#%% Data loading part 
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

np.shape(x_data)
np.shape(y_data)
#%%  Building Graph
X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 3])
W = tf.Variable(tf.random_normal([3, 3]), name = 'Weight')
b = tf.Variable(tf.random_normal([3]), name = 'bias')
#
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.5).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-10).minimize(cost)
#
prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y,1))   # because Y is one hot coded.
accuracy = tf.reduce_mean(tf.cast(is_correct,dtype=tf.float32))

#%% Launch Graph 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val , W_val, _ = sess.run([cost, W, optimizer],  feed_dict = {X:x_data, Y:y_data})
        print(step, cost_val, W_val)
        
    print("Prediction:", sess.run(prediction, feed_dict={X:x_test}))
    print("Answer:", sess.run(tf.arg_max(y_test,1)))
    print("Accuracy:", sess.run(accuracy, feed_dict={X:x_test, Y:y_test}))

#%%

