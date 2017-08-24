#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:38:45 2017

@author: Wonjoong Cheon
"""
#%%
import tensorflow as tf
import numpy as np 

#%%
sample = " My name is Wonjoong Cheon"
#sample = " if you want you"
idex2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idex2char)}

#%%
dic_size = len(char2idx)
hidden_size = len(char2idx)
num_class = len(char2idx)
batch_size = 1
sequence_length = len(sample) -1 
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

#%%
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

x_one_hot = tf.one_hot(X,num_class)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs,_states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype = tf.float32)

#
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_class, activation_fn = None)
#reshape
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_class])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#
for i in range(50):
    l, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
    result = sess.run(prediction, feed_dict={X:x_data})
    #
    result_str = [idex2char[c] for c in np.squeeze(result)]
    print(i, "loss:", l, "Prediction:", ''.join(result_str))

#%%
x_data_str = [idex2char[c] for c in np.squeeze(x_data)]
print(x_data_str)



