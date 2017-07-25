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
#W = tf.Variable(tf.random_normal([784, 256]), name = 'weight')
W = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([256]), name ='bias')
L1 = tf.nn.relu(tf.matmul(X,W)+b)
#
#W2 = tf.Variable(tf.random_normal([256, 256]), name = 'weight2')
W2 = tf.get_variable("W2", shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]), name ='bias2')
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
#
#W3 = tf.Variable(tf.random_normal([256, 10]), name ='weight3')
W3 = tf.get_variable("W3", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
hypothesis= tf.matmul(L2,W3)+b3
#
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels = Y))
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