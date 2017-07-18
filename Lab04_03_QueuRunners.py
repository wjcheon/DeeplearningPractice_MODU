#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:58:48 2017

@author: Wonjoong Cheon 
"""
#%%
import tensorflow as tf
import numpy as np 
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv','data-01-test-score.csv'])
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype = np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data, len(y_data))
#
#
X = tf.placeholder(tf.float32,shape = [None, 3])
Y = tf.placeholder(tf.float32,shape = [None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name = 'Weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
#
hypothesis = tf.matmul(X,W) + b 
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)
#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                   feed_dict = {X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)

#%%
print("Your score will be", sess.run(hypothesis, feed_dict={X:[[100, 70, 101]]}))
print("Other socre will be", sess.run(hypothesis, feed_dict={X:[[60, 70, 110], [90, 100, 80]]}))