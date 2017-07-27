#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:02:26 2017

@author: Wonjoong Cheon
"""
#%%
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
sess = tf.InteractiveSession()
image= np.array([[[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]]], dtype = np.float32)
np.shape(image)
plt.imshow(image.reshape(3,3),cmap='Greys')
print("image.shape",image.shape)

#%%
weight = tf.constant([[[[1.]],[[1.]]],[[[1.]],[[1.]]]])
print('weight.shape', weight.shape)

#%%
conv2d = tf.nn.conv2d(image,weight,strides=[1,1,1,1], padding='SAME')
conv2d_img = conv2d.eval()
print("conv2d_img.shape",conv2d_img.shape)
conv2d_img_swap = np.swapaxes(conv2d_img, 0, 3)

for i, one_img in enumerate(conv2d_img):
#for i, one_img in enumerate(conv2d_img_swap):
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
    
    #%%
print("image.shape", image.shape)
weight = tf.constant([[[[1., 10., -1.]],[[1., 10., -1.]]],[[[1., 10., -1.]],[[1., 10., -1.]]]])
print("weight.shape",weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding="SAME")
conv2d_img = conv2d.eval()
print("conv2d_img.shape",conv2d_img.shape)
conv2d_img_swap = np.swapaxes(conv2d_img, 0, 3)
print("conv2d_img_swap.shape",conv2d_img_swap.shape)
#%%
for i, one_img in enumerate(conv2d_img_swap):
    print(i, ':', one_img.shape)
    #print(one_img)
    #print(one_img.reshape(3,3))
    img_temp = one_img.reshape(3,3)
    print("np.shape(img_temp):",np.shape(img_temp))
    plt.subplot(1,3,i+1), plt.imshow(img_temp , cmap='gray')
    
    #%%
image = np.array([[[[4],[3]],[[2],[1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding = 'SAME')
print(pool.shape)
print(pool.eval())


#%%

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

img = mnist.train.images[0].reshape(28,28)
plt.imshow(img, cmap='gray')

sess = tf.InteractiveSession()
print(np.shape(img))
img = img.reshape(-1,28,28,1)
print(np.shape(img))

W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
conv2d = tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding='SAME')
print(conv2d)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img_swap = np.swapaxes(conv2d_img,0,3)
#%%
for i, one_img in enumerate(conv2d_img_swap):
    print(np.shape(one_img))
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')
    
#%%

#pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[ 1, 2, 2, 1], padding='SAME' )
pool = tf.nn.max_pool(conv2d_img, ksize=[1, 2, 2, 1], strides=[ 1, 2, 2, 1], padding='SAME' )
print(pool)
#
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7,7), cmap='gray')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    