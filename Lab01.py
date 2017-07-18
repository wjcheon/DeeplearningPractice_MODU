#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 07:58:59 2017

@author: Wonjoong Cheon
"""
#%%
import tensorflow as tf
version_tf = tf.__version__
print('The version of tensorflow is {}'.format(version_tf))

#%%
hello = tf.constant("Hello, Tensorflow!")
sess = tf.Session()
print(sess.run(hello))
#%%  Constant
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)
print("node1:", node1, "node2:", node2)
print("node3:", node3)

sess = tf.Session()
print("sess.run(node1,node2)",sess.run([node1, node2]))
print("sess.run(node3)",sess.run(node3))
results_node1 = sess.run(node1);
results_node2 = sess.run(node2);
results_node3 = sess.run(node3);

#%% Placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = tf.add(a,b)
print(sess.run((adder_node), feed_dict={a: 3,b:4.5}))
print(sess.run((adder_node), feed_dict={a:[1, 3],b:[2, 4]}))

#%%
import numpy as np
tensor_rank1 = [1., 2., 3.]
tensor_rank2 = [[1., 2., 3.,],[4.,5.,6.]]  # 2 X 3 
tensor_rank3 = [[[1., 2., 3.,], [4., 5., 6.,]]] #    1 X 2 X 3
np.shape(tensor_rank3)

tensor_3 =[[[2.], [4.], [6.]], [[2.], [4.], [6.]], [[2.], [4.], [6.]] ]
np.shape(tensor_3)

