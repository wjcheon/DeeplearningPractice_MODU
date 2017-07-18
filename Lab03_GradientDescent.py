#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:13:00 2017

@author: Wonjoong Cheon
"""
#%%
import tensorflow as tf
x_data = [1, 2, 3]
y_data = [1, 2, 3]

#W = tf.Variable(tf.random_normal([1]), name = 'weight')
W = tf.Variable(-3.0, name = 'weight')
XX = tf.placeholder(tf.float32)
YY = tf.placeholder(tf.float32)

hypothesis = XX*W
cost = tf.reduce_sum(tf.square(hypothesis - YY))

## Gradient descent(manual)
learning_rate = 0.1
gradient = tf.reduce_mean((W*XX - YY )*XX)
descent = W - gradient*learning_rate
update = W.assign(descent)
#
#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#
for step in range(30):
    sess.run(update, feed_dict={XX: x_data, YY:y_data})
    print(step, sess.run(cost,feed_dict={XX:x_data, YY:y_data}), sess.run(W))


