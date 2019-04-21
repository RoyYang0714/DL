import pandas as pd
import numpy as np 
import tensorflow as tf
import sys
sys.path.append('source')


def compute_accuracy(v_xs, v_ys):
	global prediction

	y_pre = sess.run(prediction, feed_dict={xs:v_xs})
	correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs:X_valid, ys:Y_valid})

	return result

### data processing and loading ###

import data
data_train, data_valid = data.spilt()

x_train = data_train.drop(['dws', 'ups', 'sit', 'std', 'wlk', 'jog'], axis=1)
x_valid = data_valid.drop(['dws', 'ups', 'sit', 'std', 'wlk', 'jog'], axis=1)
y_train = data_train.loc[:, ['dws', 'ups', 'sit', 'std', 'wlk', 'jog']]
y_valid = data_valid.loc[:, ['dws', 'ups', 'sit', 'std', 'wlk', 'jog']]

X_train = x_train.values
X_valid = x_valid.values
Y_train = y_train.values
Y_valid = y_valid.values

X_train = X_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
Y_train = Y_train.astype(np.float32)
Y_valid = Y_valid.astype(np.float32)

### HYPERPARAMETERS ###
epochs = 1500
mini_batch_size = 32
learning_rate = 0.0001

### construct nn ###
import layer

with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 68], name = 'x_train')
	ys = tf.placeholder(tf.float32, [None, 6],  name = 'y_train')
 
l1 = layer.add_layer(xs, 68, 30, n_layer = '1', activation_function=tf.nn.relu)
l2 = layer.add_layer(l1, 30, 15, n_layer = '2', activation_function=tf.nn.relu)
prediction = layer.add_layer(l2, 15, 6, n_layer = '3', activation_function=tf.nn.softmax)

with tf.name_scope('loss'):
	loss = tf.multiply(tf.reduce_mean(tf.multiply(ys, tf.log(prediction))), -1.)
	tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

init = tf.initialize_all_variables()

### activate tensorflow ###
with tf.Session() as sess:
	sess.run(init)

	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter("logs/", sess.graph)	

	for epoch in range(1500):
		'''
		x = X_train[np.random.randint(0,X_train.shape[0],mini_batch_size)]
		y = Y_train[np.random.randint(0,X_train.shape[0],mini_batch_size)]
		sess.run(train_step, {xs:x, ys:y})
		'''
		for i in range(mini_batch_size):
			x = X_train[(i+epoch*mini_batch_size)%4406].reshape((1,68))
			y = Y_train[(i+epoch*mini_batch_size)%4406].reshape((1,6))
			sess.run(train_step, {xs:x, ys:y})
		

		print('epoch:', epoch, 'accuracy:', compute_accuracy(X_valid, Y_valid), 'loss:', sess.run(loss, {xs:x, ys:y}))						
		result = sess.run(merged, feed_dict={xs:X_train, ys:Y_train})
		writer.add_summary(result, epoch)
