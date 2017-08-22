#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:02:39 2017

@author: Wonjoong Cheon
"""


#%%
import tensorflow as tf
import numpy as np 
from tensorflow.contrib import rnn 
import pprint 
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

#%%
h = [1, 0, 0, 0]
e = [1, 1, 0, 0]
l = [1, 0, 1, 0]
o = [1, 0, 0, 1]

#%% 01 
with tf.variable_scope('one_cell') as scope:
    hidden_size = 2  # the number of neuron
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    print(cell.output_size, cell.state_size)
    #
    x_data = np.array([[h]], dtype=np.float32)
    pp.pprint(x_data)
    x_data.shape
    outputs, states_ = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
    print("shape of outputs:", outputs.shape)
    print("shape of outputs:", states_.shape)
    
#%% 02 
with tf.variable_scope('one_cell') as scope:
    hidden_size = 2  # the number of neuron
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    print(cell.output_size, cell.state_size)
    #
    x_data = np.array([[h,e,l,l,o]], dtype=np.float32)
    pp.pprint(x_data)
    x_data.shape
    outputs, states_ = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
    print("shape of outputs:", outputs.shape)
    print("shape of outputs:", states_.shape)
#%% 03
with tf.variable_scope('one_cell') as scope:
    hidden_size = 2  # the number of neuron
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    print(cell.output_size, cell.state_size)
    #
    x_data = np.array([[h,e,l,l,o],
                       [e,o,l,l,l],
                       [l,l,e,e,l]], dtype=np.float32)
    pp.pprint(x_data)
    x_data.shape
    outputs, states_ = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
    print("shape of outputs:", outputs.shape)
    print("shape of outputs:", states_.shape)
    
#%% 04
with tf.variable_scope('3_batches_dynamic_length') as scope:
    hidden_size = 2  # the number of neuron
    #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple = True)
    print(cell.output_size, cell.state_size)
    #
    x_data = np.array([[h,e,l,l,o],
                       [e,o,l,l,l],
                       [l,l,e,e,l]], dtype=np.float32)
    pp.pprint(x_data)
    x_data.shape
    #
    #outputs, states_ = tf.nn.dynamic_rnn(cell, x_data, sequence_length=[5,3,4], dtype=tf.float32)
    outputs, states_ = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
    print("shape of outputs:", outputs.shape)
    
#%%
with tf.variable_scope('initial_state') as scope:
    batch_size = 3
    hidden_size = 2  # the number of neuron
    #
    x_data = np.array([[h,e,l,l,o],
                       [e,o,l,l,l],
                       [l,l,e,e,l]], dtype=np.float32)
    pp.pprint(x_data)
    x_data.shape
    #
    #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple = True)
    print(cell.output_size, cell.state_size)
    #
    initial_state = cell.zero_state(batch_size, tf.float32)
    print(initial_state)
    outputs, states_ = tf.nn.dynamic_rnn(cell, x_data, initial_state=initial_state, dtype=tf.float32)
    
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
    outputs.shape
    print("shape of outputs:", outputs.shape)
    
#%% Create input data
batch_size = 3
sequence_length = 5
input_dim = 3
hidden_dim = 5

x_data =np.arange(45, dtype = np.float32).reshape(batch_size, sequence_length, input_dim)
pp.pprint(x_data)
print(x_data.shape)

with tf.variable_scope('generated_data') as scope:
    cell = rnn.BasicLSTMCell(num_units= hidden_dim, state_is_tuple = True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, initial_state=initial_state, dtype= tf.float32)
    #
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
    print(outputs.shape)
    
#%%
batch_size = 3
sequence_length = 5
input_dim = 3
hidden_dim = 5

x_data =np.arange(45, dtype = np.float32).reshape(batch_size, sequence_length, input_dim)
pp.pprint(x_data)
print(x_data.shape)

with tf.variable_scope('MultiRNNcekk') as scope:
    #cell_single = rnn.BasicLSTMCell(num_units=hidden_dim,  state_is_tuple= True)
    #cell = rnn.MultiRNNCell([cell_single]*3, state_is_tuple = True)
    cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units=hidden_dim,  state_is_tuple= True) for _ in range(3)], state_is_tuple = True)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype= tf.float32)
    print("dynamic rnn (multi*3): ", outputs)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
    outputs.shape


#%%
batch_size = 3
sequence_length = 5
input_dim = 3
hidden_dim = 5
#
x_data =np.arange(45, dtype = np.float32).reshape(batch_size, sequence_length, input_dim)
pp.pprint(x_data)
print(x_data.shape)
#
with tf.variable_scope('dynamic_length_rnn') as scope:
    cell =  rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data,dtype=tf.float32, sequence_length=[1,3,2])
    #
    print("dynamic rnn: ", outputs)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
    
#%%
batch_size = 3
sequence_length = 5
input_dim = 3
hidden_dim = 5
#
x_data =np.arange(45, dtype = np.float32).reshape(batch_size, sequence_length, input_dim)
pp.pprint(x_data)
print(x_data.shape)
#
with tf.variable_scope('bi-directional') as scope:
    cell_fw = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple = True)
    cell_bw = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple = True)
    #
    outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_data, sequence_length=[2,3,1], dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    output_val = sess.run(outputs)
    states_val = sess.run(_states)
    pp.pprint(sess.run(outputs))
    pp.pprint(sess.run(_states)) 

#%%
hidden_size = 3
sequence_length = 5
batch_size = 3
num_classes = 5
#
x_data =np.arange(45, dtype = np.float32).reshape(batch_size, sequence_length, input_dim)
print(x_data.shape)
x_data = x_data.reshape(-1, hidden_size)
print(x_data.shape)
#
softmax_w = np.arange(15, dtype = np.float32).reshape(hidden_size, num_classes)
outputs = np.matmul(x_data,softmax_w)
outputs = outputs.reshape(-1, sequence_length, num_classes)
pp.pprint(outputs)
#
y_data =  tf.constant([[1], [1], [1]])
#
prediction = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.3, 0.7]]], dtype = tf.float32)
#
weights = tf.constant([[1,1,1]], dtype = tf.float32)
#
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = prediction, targets= y_data, weights = weights)
sess.run(tf.global_variables_initializer())
print("Loss: ", sequence_loss.eval())

#%%
hidden_size = 3
sequence_length = 5
batch_size = 3
num_classes = 5
#
x_data =np.arange(45, dtype = np.float32).reshape(batch_size, sequence_length, input_dim)
print(x_data.shape)
x_data = x_data.reshape(-1, hidden_size)
print(x_data.shape)
#
softmax_w = np.arange(15, dtype = np.float32).reshape(hidden_size, num_classes)
outputs = np.matmul(x_data,softmax_w)
outputs = outputs.reshape(-1, sequence_length, num_classes)
pp.pprint(outputs)
#
y_data =  tf.constant([[1, 1, 1]])
#
prediction1 = tf.constant([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]], dtype = tf.float32)
prediction2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype = tf.float32)
prediction3 = tf.constant([[[1, 0], [1, 0], [1, 0]]], dtype = tf.float32)
prediction4 = tf.constant([[[0, 1], [1, 0], [0, 1]]], dtype = tf.float32)
#
weights = tf.constant([[1,1,1]], dtype = tf.float32)
#
sequence_loss1 = tf.contrib.seq2seq.sequence_loss(logits = prediction1, targets= y_data, weights = weights)
sequence_loss2 = tf.contrib.seq2seq.sequence_loss(logits = prediction2, targets= y_data, weights = weights)
sequence_loss3 = tf.contrib.seq2seq.sequence_loss(logits = prediction3, targets= y_data, weights = weights)
sequence_loss4 = tf.contrib.seq2seq.sequence_loss(logits = prediction4, targets= y_data, weights = weights)

sess.run(tf.global_variables_initializer())
print("Loss1: ", sequence_loss1.eval(),
      "Loss2: ", sequence_loss2.eval(),
      "Loss3: ", sequence_loss3.eval(),
      "Loss4: ", sequence_loss4.eval())




#%%









































