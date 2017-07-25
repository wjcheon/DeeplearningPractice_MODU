#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:15:43 2017

@author: Wonjoong Cheon
"""
#%% Lab 7 Learning rate and Evaluation
import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
#
W = tf.Variable(tf.random_normal([784, 10]), name = 'weight')
b = tf.Variable(tf.random_normal([10]), name ='bias')
#
hypothesis = tf.matmul(X,W)+b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels = Y))
#
#
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 15
batch_size = 100
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples /batch_size)
    #
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs, Y:batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch 
    print('Epch:', '%04d' % (epoch +1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning finished')
#
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

#aa = [[0.1], [0.8], [0.1] ]
#aa2 = [0.1, 0.8, 0.1 ]
#import numpy as np
#np.shape(aa2)
#np.shape(aa)
#print(sess.run(tf.arg_max(aa,0)))
#test_argmax = sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images, Y:mnist.test.labels})
#test_arg_max = sess.run(tf.arg_max(hypothesis, 1), feed_dict={X:mnist.test.images, Y:mnist.test.labels})
#
#    


