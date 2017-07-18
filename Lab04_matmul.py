#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 12:50:12 2017

@author: Wonjoong Cheon
"""
#%%
import tensorflow as tf
import numpy as np
#x1_data = [73., 93., 89., 96., 73.]
#x2_data = [80., 88., 91., 98., 66.]
#x3_data = [75., 93., 90., 100., 70.]
#y_data = [152., 185., 180., 196., 142.]
#
#
x_data = [[73., 80., 75.],[93., 88., 93.],
          [89., 91., 90.],[96., 98., 100.], [73.,66.,70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]
np.shape(x_data)
#
#
#x1 = tf.placeholder(tf.float32)
#x2 = tf.placeholder(tf.float32)
#x3 = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, shape =[None, 3])
Y= tf.placeholder(tf.float32, shape = [None, 1])
#
#
#w1 = tf.Variable(tf.random_normal([1]),name ='Weight1')
#w2 = tf.Variable(tf.random_normal([1]),name ='Weight2')
#w3 = tf.Variable(tf.random_normal([1]),name ='Weight3')
#b = tf.Variable(tf.random_normal([1]),name ='bias')
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
#
#hypothesis = x1*w1 + x2*w2 + x3*w3 + b
hypothesis = tf.matmul(X,W) + b
#
cost  = tf.reduce_mean(tf.square(hypothesis - Y))
#
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)
#
#
#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    #cost_val, hypothesis_val, _ = sess.run([cost, hypothesis, train], feed_dict ={x1: x1_data, x2:x2_data, x3: x3_data, Y:y_data})
    cost_val, hypothesis_val, _ = sess.run([cost, hypothesis, train], feed_dict ={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hypothesis_val)