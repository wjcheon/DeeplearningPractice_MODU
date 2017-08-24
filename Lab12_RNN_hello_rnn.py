#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:24:28 2017

@author: Wonjoong Cheon
"""

#%%
import tensorflow as tf 
import numpy as np
tf.set_random_seed(777)

idex2char = ['h', 'i', 'e', 'l', 'o']
# Teach hello: hihell -> ihello 
x_data = [[[0,1,0,2,3,3]]]     # hihell
x_one_hot = [[[1,0,0,0,0],
              [0,1,0,0,0],
              [1,0,0,0,0],
              [0,0,1,0,0],
              [0,0,0,1,0],
              [0,0,0,1,0]]]
x_data2 = [[0,1,0,2,3,3]]

y_data = [[1,0,2,3,3,4]]    # ihello

#%%
num_classes = 5 
input_dim = 5      #one_hot_size , the number of feature
hidden_size = 5    # output from the LSTM. the number of neuron in LSTM
batch_size = 1     # 1
sequence_length = 6     # the number of time sequence 
learning_rate = 0.1     # learning rate for learning 

X = tf.placeholder(tf.float32, [None,  sequence_length,  input_dim])
Y = tf.placeholder(tf.int32, [None, sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple= True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype= tf.float32)
print("output_size:", cell.output_size)
# FC layer
X_for_fc = tf.reshape(outputs,[-1, hidden_size])
outputs_old = outputs
outputs = tf.contrib.layers.fully_connected(inputs=X_for_fc, num_outputs= num_classes, activation_fn=None)

#
outputs_old2 = outputs
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights = weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

#%%
sess= tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(50):
    l, _ = sess.run([loss, train], feed_dict={X:x_one_hot, Y:y_data})
    result = sess.run(prediction, feed_dict={X: x_one_hot})
    print(i, "loss", l, "prediction:", result, "True Y:", y_data )
    #
    result_str = [idex2char[c] for c in np.squeeze(result)]
    print("\tPrediction str: ", "".join(result_str))

        


#%%
#sequence_loss2 = tf.contrib.seq2seq.sequence_loss(logits=X, targets=Y, weights = weights)
#sess.run(sequence_loss, feed_dict={X:x_one_hot, Y:y_data})