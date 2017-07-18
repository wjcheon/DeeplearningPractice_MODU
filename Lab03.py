#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:13:01 2017

@author: Wonjoong Cheon 
"""
#%%
import tensorflow as tf
import matplotlib.pyplot as plt 
#%%
X = [1., 2., 3.]
Y = [1., 2., 3.]

W = tf.placeholder(tf.float32)
hypothesis = W*X
cost = tf.reduce_mean(tf.square(hypothesis-Y))
#
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#
W_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W],feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)
    
#%%
plt.plot(W_val, cost_val)
plt.xlabel('W'), plt.ylabel('cost')
plt.grid()
plt.show()
