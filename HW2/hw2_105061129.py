import pandas as pd
import numpy as np 
import tensorflow as tf
import sklearn as sk
from sklearn.metrics import confusion_matrix
import sys
sys.path.append('source')

### data processing and loading ###
import data
data_train, data_valid = data.spilt()

x_train = data_train.drop(['Activities_Types', 'dws', 'ups', 'sit', 'std', 'wlk', 'jog'], axis=1)
x_valid = data_valid.drop(['Activities_Types', 'dws', 'ups', 'sit', 'std', 'wlk', 'jog'], axis=1)
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
learning_rate1 = 0.001
learning_rate2 = 0.2
learning_rate3 = 0.1

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
	tf.summary.scalar('train', loss)

with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(learning_rate1).minimize(loss)
	#train_step = tf.train.GradientDescentOptimizer(learning_rate2).minimize(loss)
	#train_step = tf.train.AdagradOptimizer(learning_rate3).minimize(loss)

with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

init = tf.initialize_all_variables()

### activate tensorflow ###
with tf.Session() as sess:
	sess.run(init)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter("logs/train", sess.graph)
	validate_writer = tf.summary.FileWriter("logs/validate")

	for epoch in range(epochs):
		flag = 0
		x = np.zeros((1, 68))
		y = np.zeros((1, 6))
		for i in range(mini_batch_size):
			if flag != 0:
				x = X_train[(i+epoch*mini_batch_size)%4406].reshape((1,68))
				y = Y_train[(i+epoch*mini_batch_size)%4406].reshape((1,6))
				flag = 1
			else:
				x = np.r_[x, X_train[(i+epoch*mini_batch_size)%4406].reshape((1,68))]
				y = np.r_[y, Y_train[(i+epoch*mini_batch_size)%4406].reshape((1,6))]
		sess.run(train_step, {xs:x, ys:y})
		
		print('epoch:', epoch, 'accuracy:', sess.run(accuracy, {xs:X_valid, ys:Y_valid}), 'loss:', sess.run(loss, {xs:X_valid, ys:Y_valid}))						
		
		result = sess.run(merged, feed_dict={xs:X_train, ys:Y_train})
		train_writer.add_summary(result, epoch)

		result = sess.run(merged, feed_dict={xs:X_valid, ys:Y_valid})
		validate_writer.add_summary(result, epoch)

	y_p = tf.argmax(sess.run(prediction, feed_dict={xs:X_valid, ys:Y_valid}), 1)
	y_p = sess.run(y_p)
	y_true = np.argmax(Y_valid, 1)

	print("micro-Precision", sk.metrics.precision_score(y_true, y_p, average=None))
	print("macro-Precision", sk.metrics.precision_score(y_true, y_p, average='macro'))
	print("micro-Recall", sk.metrics.recall_score(y_true, y_p, average=None))
	print("macro-Recall", sk.metrics.recall_score(y_true, y_p, average='macro'))
	print("micro-F1_score", sk.metrics.f1_score(y_true, y_p, average=None))
	print("macro-F1_score", sk.metrics.f1_score(y_true, y_p, average='macro'))

	data_test = pd.read_csv('data/Test_no_Ac.csv')

	data_test = data_test.values
	data_test = data_test.astype(np.float32)

	p = tf.argmax(sess.run(prediction, feed_dict={xs:data_test}), 1) + 1
	p = sess.run(p)

	p = p.reshape((1378, 1))

	i = np.arange(1,1379,1)
	i = np.transpose(i).reshape((1378, 1))

	p = np.c_[i, p]

	df = pd.DataFrame({'Column1':p[:,0],'Column2':p[:,1]})
	np.savetxt(r"105061129_answer.txt", df.values, fmt='%d	%d')

### PCA ###
import PCA

PCA.plot(data_valid)

### t-SNE ###
import t_SNE

t_SNE.plot(data_valid)
