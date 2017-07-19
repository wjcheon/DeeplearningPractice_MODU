# Lab 4 Multi-variable linear regression
# https://www.tensorflow.org/programmers_guide/reading_data
#%%
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

#ilename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name='filename_queue')
filename_queue = tf.train.string_input_producer(['data-03-diabetes.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=700)
    
    
    #%%
#sess2 = tf.Session()
#coord = tf.train.Coordinator()
#threads = tf.train.start_queue_runners(sess=sess2, coord=coord)
#x_batch_test = sess2.run(train_x_batch)
#y_batch_test = sess2.run(train_y_batch)
#print(x_batch_test)
#%%
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# Simplified cost/loss function
#cost = tf.reduce_mean(tf.square(hypothesis - Y))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1- hypothesis))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
train = optimizer.minimize(cost)
#
predicted = tf.cast(hypothesis > 0.5 , dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype= tf.float32))
#%%
# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(10001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    #print(x_batch)
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 20 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
#     
coord.request_stop()
coord.join(threads)
#
#
#%%
import numpy as np
xy_validation = np.loadtxt('data-03-diabetes.csv',delimiter=',',dtype=np.float32)
x_data_validation = xy_validation [:,0:-1]
y_data_validation = xy_validation [:,[-1]]
feed_validation = {X:x_data_validation, Y:y_data_validation}
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = feed_validation)
print("Hypotehsis:", h)
print("Corrected:", c)
print("Accuracy:", a)
