#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:48:23 2017

@author: Wonjoong Cheon
"""
#%%
import tensorflow as tf 
X = [1., 2., 3.]
Y = [1., 2., 3.]
#
W = tf.Variable(5.0, name='Weight')
#
hypothesis = X*W
#
gradient = tf.reduce_mean((W*X-Y)*X) *2 
#
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
#
#
gvs = optimizer.compute_gradients(cost, [W])
#
apply_gradient = optimizer.apply_gradients(gvs)
#
#
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#
#%%
for step in range(100):
    print(step, sess.run([gradient,gvs,W]))
    sess.run(apply_gradient)
