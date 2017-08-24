#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:10:01 2017

@author: Wonjoong Cheon
"""
#%%
import tensorflow as tf
a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2,3], name='a')
b = tf.constant([1., 2., 3., 4., 5., 6.], shape=[3,2], name='b')
c = tf.matmul(a,b)
# Creates a session with log_device_placement
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op. 
print(sess.run(c))
#%%
with tf.device('/cpu:0'):
    a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2,3], name='a')
    b = tf.constant([1., 2., 3., 4., 5., 6.], shape=[3,2], name='b')
c= tf.matmul(a,b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op. 
print(sess.run(c))

#%%
with tf.device('/gpu:0'):
    a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2,3], name='a')
    b = tf.constant([1., 2., 3., 4., 5., 6.], shape=[3,2], name='b')
    c= tf.matmul(a,b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op. 
print(sess.run(c))
#%% Multi GPU tower 
import tensorflow as tf

c = []
for d in ['/gpu:0', '/gpu:1']:
    with tf.device(d):
        a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2,3], name='a')
        b = tf.constant([1., 2., 3., 4., 5., 6.], shape=[3,2], name='b')
        c.append(tf.matmul(a,b))
with tf.device('/cpu:0'):
    sum = tf.add_n(c)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op. 
print(sess.run(c))


