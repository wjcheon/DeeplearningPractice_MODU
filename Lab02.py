#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:26:14 2017

@author: Wonjoong Cheon
"""
#%%
import tensorflow as tf 
import numpy as np
tf.set_random_seed(777) # for reproducibility
#x_train = [1., 2., 3.];
#y_train = [1., 2., 3.];
#x_train = tf.placeholder(tf.float32)
#y_train = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, shape =[None])
Y = tf.placeholder(tf.float32, shape =[None])

#%%
W = tf.Variable(tf.random_normal([1]), name = 'weight') 
b = tf.Variable(tf.random_normal([1]), name = 'bias')
hypothesis = X *W + b 
#
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
#
train  = optimizer.minimize(cost)
#
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#%%
for step in range(2000):
    sess.run(train,feed_dict={X: [1., 2., 3.], Y:[1., 2., 3.]})
    #cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],feed_dict={X : [1., 2., 3.], Y:[1.,2.,3.]})
    #cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
#%%
for step in range(2001):
    _, cost_val = sess.run([train, cost], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b), cost_val)