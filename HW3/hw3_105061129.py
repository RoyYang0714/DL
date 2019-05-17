import numpy as np 
import tensorflow as tf
import data_prepro
import math
import matplotlib.pyplot as plt

### HYPERPARAMETERS ###
epochs = 50
mini_batch_size = 32
learning_rate = 1e-4
size = 128
norm = 0
drop_out = 1

### build CNN layer ###
def batch_norm(Wx_plus_b, out_size):
	if norm == 1:
		mean, var = tf.nn.moments(Wx_plus_b, axes=[0, 1, 2])
		scale = tf.Variable(tf.ones([out_size]))
		shift = tf.Variable(tf.zeros([out_size]))

		return tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, 0.001)
	else:
		return Wx_plus_b

xs = tf.placeholder(tf.float32, [None, size, size, 3])
ys = tf.placeholder(tf.float32, [None, 101])
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('conv1'):
	W_conv1 = tf.Variable(tf.random.truncated_normal([3, 3, 3, 16], stddev=0.05))
	b_conv1 = tf.Variable(tf.constant(0.0, shape=[16])) 
	logist1 = tf.nn.conv2d(xs, W_conv1, strides=[1,1,1,1], padding='SAME')+ b_conv1
	h_norm1 = batch_norm(logist1, 16)
	h_conv1 = tf.nn.relu(h_norm1)
	h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	tf.summary.histogram('Histogram of conv1', W_conv1)


with tf.name_scope('conv2'):
	W_conv2 = tf.Variable(tf.random.truncated_normal([3, 3, 16, 32], stddev=0.05))
	b_conv2 = tf.Variable(tf.constant(0.0, shape=[32]))
	logist2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME')+ b_conv2
	h_norm2 = batch_norm(logist2, 32)
	h_conv2 = tf.nn.relu(h_norm2)
	h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	tf.summary.histogram('Histogram of conv2', W_conv2)

with tf.name_scope('conv3'):
	W_conv3 = tf.Variable(tf.random.truncated_normal([3, 3, 32, 64], stddev=0.05))
	b_conv3 = tf.Variable(tf.constant(0.0, shape=[64]))
	logist3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1,1,1,1], padding='SAME')+ b_conv3
	h_norm3 = batch_norm(logist3, 64)
	h_conv3 = tf.nn.relu(h_norm3)
	h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	tf.summary.histogram('Histogram of conv3', W_conv3)

size = h_pool3.get_shape()
flatten = tf.reshape(h_pool3, [-1, size[1]*size[2]*size[3]])
input_size = int(flatten.get_shape()[1])

with tf.name_scope('fc1'):
	W_fc1 = tf.Variable(tf.random.truncated_normal([input_size, 1024], stddev=0.05))
	b_fc1 = tf.Variable(tf.constant(0.0, shape=[1024]))
	h_fc1 = tf.nn.relu(tf.matmul(flatten, W_fc1) + b_fc1)
	tf.summary.histogram('Histogram of fc1', W_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
	W_fc2 = tf.Variable(tf.random.truncated_normal([1024, 101], stddev=0.05))
	b_fc2 = tf.Variable(tf.constant(0.0, shape=[101]))
	logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	prediction = tf.nn.softmax(logits)
	tf.summary.histogram('Histogram of fc2', W_fc2)

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=logits))
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=logits)) + tf.nn.l2_loss(prediction)
	tf.summary.scalar('loss', loss)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'

with tf.Session(config = config) as sess:
	sess.run(tf.global_variables_initializer())

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter("logs/train", sess.graph)
	validate_writer = tf.summary.FileWriter("logs/validate", sess.graph)

	X, Y , label_names = data_prepro.data('train')
	x_t, y_t, label_names = data_prepro.data('test')

	for epoch in range(epochs):
		batch_iter = int(math.ceil(len(X)/mini_batch_size))

		for i in range(batch_iter):
			start = i*mini_batch_size
			end = min((i+1)*mini_batch_size,len(X))
			x = X[start:end]
			y = Y[start:end]

			sess.run(train_step, feed_dict={xs:x, ys:y, keep_prob:drop_out})

		#x = tf.image.rot90(x, k = epoch%4+1).eval()

		#sess.run(train_step, feed_dict={xs:x, ys:y, keep_prob:drop_out})

		result = sess.run(merged, feed_dict={xs:x, ys:y, keep_prob:1})
		train_writer.add_summary(result, epoch)

		result = sess.run(merged, feed_dict={xs:x_t, ys:y_t, keep_prob:1})
		validate_writer.add_summary(result, epoch)

		print('epoch:',epoch ,'accuracy:', sess.run(accuracy, feed_dict={xs:x_t, ys:y_t, keep_prob:1}), 'loss:', sess.run(loss, feed_dict={xs:x_t, ys:y_t, keep_prob:1}))

	pred = tf.cast(tf.argmax(sess.run(prediction, feed_dict={xs:x_t, ys:y_t, keep_prob:1}), 1), tf.float32)
	pred = label_names[int(pred[0].eval())]

	for  i in range(101):
		if y_t[0][i] == 1:
			l = label_names[i]

	plt.imshow(x_t[0])
	plt.title("pred: %s, true label: %s" %(pred ,l))
	plt.show()

	x_conv1 = sess.run(h_conv1, feed_dict={xs:x_t, ys:y_t, keep_prob:1})

	plt.figure()
	for i in range(16):
		plt.subplot(4,4,i+1)
		plt.imshow(x_conv1[0][:,:,i])
		plt.axis('off')
	plt.show()	

	x_conv2 = sess.run(h_conv2, feed_dict={xs:x_t, ys:y_t, keep_prob:1})

	plt.figure()
	for i in range(16):
		plt.subplot(4,4,i+1)
		plt.imshow(x_conv2[0][:,:,i])
		plt.axis('off')
	plt.show()
