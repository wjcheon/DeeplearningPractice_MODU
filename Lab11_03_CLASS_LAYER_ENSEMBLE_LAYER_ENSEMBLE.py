#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:08:31 2017

@author: Wonjoong Cheon
"""
#%%
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
#%%
class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        #self._build_net()
        
    def set_paramerters(self):
        self.keep_prob = tf.placeholder(tf.float32)
        
        
    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            #
            self.training = tf.placeholder(tf.bool)
            #
            #
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size = [3,3], padding="SAME",activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,rate=self.keep_prob, training=self.training )
            print(dropout1)
        
            ##
            #L2 = tf.reshape(L2, [-1, 7* 7* 64])
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size = [3,3], padding="SAME",activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=self.keep_prob, training=self.training )
            print(dropout2)         
            ##
            ##            
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size = [3,3], padding="SAME",activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=self.keep_prob, training=self.training )
            print(dropout3)    
            # fully connected 
            flat = tf.reshape(dropout3, [-1, 4* 4* 128])
            print(flat)
            ##
            dense4 = tf.layers.dense(inputs = flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,rate=self.keep_prob, training=self.training )
                                  
            ##
            #W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            #b5 = tf.Variable(tf.random_normal([10]))
            #dense5 = tf.layers.dense(inputs=dropout4, units=10)
            dense5 = tf.layers.dense(inputs=dropout4, units=10)                    
                      

            self.hypothesis = dense5
            
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels = self.Y))
            self.learning_rate = 0.001
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            self.correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            
            
            
    def predict(self, x_test):
        return self.sess.run(self.hypothesis, feed_dict={self.X:x_test, self.keep_prob: 1.0, self.training:False})
    
    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X:x_test, self.Y:y_test, self.keep_prob:1.0, self.training:False})
    
    def train(self, x_test, y_test):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X:x_test, self.Y:y_test, self.keep_prob:0.7, self.training:True})
    
#%%
sess = tf.Session()
models = []
num_models = 7
for m in range(num_models):
    models.append(Model(sess,"model" + str(m)))
#
for mm in range(num_models):
    models[mm].set_paramerters()
    models[mm]._build_net()
#
sess.run(tf.global_variables_initializer())
print('Learning stated!!')

learning_rate = 0.001
training_epochs = 15
batch_size = 100


#%%

for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)
    #
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #
        for m_idx, model_single in enumerate(models):
            c, _ = model_single.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch
        print("Epoch:", '%04d' % (epoch+1), 'cost =', avg_cost_list)
print("Learning Finished")
#%%
#%%
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size* 10).reshape(test_size,10)

for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(mnist.test.labels,1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

#%%












































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            