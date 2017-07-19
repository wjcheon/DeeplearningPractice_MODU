#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 08:09:50 2017

@author: Wonjoong Cheon
"""
#%%
import tensorflow as tf 
import numpy as np
#%% Data load 
xy = np.loadtxt('data-03-diabetes.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]
#%% Buling Graph
X = tf.placeholder(dtype=tf.float32, shape=[None, 8])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([8, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
#
# Hypothesis
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
# cost 
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1- hypothesis))
# train
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
#
# Predicted , Accuracy 
predicted = tf.cast(hypothesis > 0.5 , dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype= tf.float32))
#%% Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    feed = {X:x_data, Y:y_data}
    for step in range(20001):
        sess.run(train, feed_dict = feed)
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict=feed))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = feed)
    print("Hypotehsis:", h)
    print("Corrected:", c)
    print("Accuracy:", a)

