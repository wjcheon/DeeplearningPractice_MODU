#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:09:36 2017

@author: Wonjoong Cheon
"""
#%%
import numpy as np 
import tensorflow as tf
import numpy as np
import pprint
tf.set_random_seed(777)  # for reproducibility\n",

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t)
print(t.ndim)
print(t.shape)
#%%
t = tf.constant([1,2,3,4])
tf.shape(t).eval()
#%%
t = tf.constant([[1,2],[3,4]])
tf.shape(t).eval()

#%%
t = [[[[4,3,2,1],[4,3,2,1],[4,3,2,1]],[[4,3,2,1],[4,3,2,1],[4,3,2,1]]]]
tf.shape(t).eval()

#%%
matrix_x1 = tf.constant([[1., 2.], [3., 4.]])
#matrix_x2 = tf.constant([[1,2]])
matrix_x2 = tf.constant([[1.],[2.]])
print("Metrix 1 shape" , matrix_x1.shape)
print("Metrix 1 shape" , matrix_x2.shape)
tf.matmul(matrix_x1,matrix_x2).eval()

#%% ERROR
(matrix_x1*matrix_x2).eval()
#%%
tf.reduce_mean([1,2], axis=0).eval()
tf.reduce_mean([1.,2.], axis = 0).eval()

x = [[1., 2.],[3., 4.]]
tf.reduce_mean(x, axis =0).eval()
tf.reduce_mean(x, axis =1).eval()
tf.reduce_mean(x, axis =-1).eval()
#%%
tf.reduce_sum(x).eval()
tf.reduce_sum(x, axis =0).eval()
tf.reduce_sum(x, axis =1).eval()
tf.reduce_sum(x,axis=-1).eval()
tf.reduce_mean(tf.reduce_sum(x,axis=-1)).eval()

#%%
t = np.array([[[0, 1, 2],[3,4,5]],[[6,7,8],[9,10,11]]])
print(t.shape)

tf.reshape(t, shape=[ -1, 3]).eval()
tf.reshape(t, shape=[-1, 1, 3]).eval()

#%%
xx = np.array([[0], [1], [2]])
print(xx.shape)
xx_squeeze = tf.squeeze(xx).eval()
print(xx_squeeze.shape)
xx_squeeze_expand_dims = tf.expand_dims(xx_squeeze,1).eval()
print(xx_squeeze_expand_dims.shape)

#%%
xx = np.array([[0], [1], [2], [0]])
ont_hot_raw = tf.one_hot(xx,depth=3).eval()
ont_hot_raw_reshape = tf.reshape(ont_hot_raw, shape=[-1 , 3]).eval()

#%% Casting
tf.cast([1.8, 2.2 , 3.3 , 4.9], tf.int32).eval()
tf.cast([True, False, 1==1, 1==0], tf.int32).eval()

#%% Stack 
x = [1, 4]
y = [2, 5]
z = [3, 6]
list_stack = [x,y,z]
np.shape(list_stack)
tensorflow_stack = tf.stack([x,y,z]).eval()
np.shape(tensorflow_stack)
tensorflow_stack_axis1 = tf.stack([x,y,z], axis = 1).eval()
np.shape(tensorflow_stack_axis1)

#%%
x = [[0, 1, 2,],[2,1,0]]
tf.ones_like(x).eval()
zeros_like_output = tf.zeros_like(x).eval()
zeros_like_output.shape
#%%
for x, y, z in zip([1,2,3], [4,5,6], [7,8,9]):
    print(x,y,z)










