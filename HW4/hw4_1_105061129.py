import tensorflow as tf
import numpy as np
from hw4_1_preprocess import *
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

### HYPERPARAMETERS ###
epochs = 10
batch_size = 125
learning_rate = 1e-4
hidden_units = 128
embedding_dim = 10000
seq_len = 256
### embedding layer ####
xs = tf.placeholder(tf.int64, [None, seq_len])

embeddings_matrix = embedding=tf.Variable(np.identity(embedding_dim ,dtype=np.float32))
	
word_embeddedings = tf.nn.embedding_lookup(embeddings_matrix, xs)

### RNN ###
ys = tf.placeholder(tf.float32, [None, 1])

weights = {
	'in':tf.Variable(tf.random.truncated_normal(shape=[embedding_dim, hidden_units], stddev=0.1)),
	'out':tf.Variable(tf.random.truncated_normal(shape=[hidden_units, 1], stddev=0.1))
}

biases = {
	'in':tf.Variable(tf.constant(0.1, shape=[hidden_units, ])),
	'out':tf.Variable(tf.constant(0.1, shape=[1, ]))
}

X = tf.reshape(word_embeddedings, [-1, embedding_dim])
X_in = tf.matmul(X, weights['in']) + biases['in']
X_in = tf.reshape(X_in, [-1, seq_len, hidden_units])

#cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_units)
#cell = tf.contrib.rnn.GRUCell(num_units = hidden_units)
cell = tf.contrib.rnn.LSTMCell(num_units = hidden_units, forget_bias=1.0, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)

#outputs, state = tf.nn.dynamic_rnn(cell, X_in, initial_state = initial_state)
outputs, state = tf.nn.dynamic_rnn(cell, X_in, initial_state = initial_state, time_major=False)

outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))

logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
prediction = tf.cast(tf.sigmoid(logits)>0.5, tf.float32)

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=ys))
	tf.summary.scalar('loss', loss)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(ys, prediction)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter("logs/train", sess.graph)
	validate_writer = tf.summary.FileWriter("logs/validate", sess.graph)

	### data loading ###
	data_train, data_test, label_train, label_test = data_load(seq_len)
	label_train = label_train.reshape((25000, 1))
	label_test = label_test.reshape((25000, 1))

	for epoch in range(epochs):
		batch_iter = int(math.ceil(25000/batch_size))

		for i in range(batch_iter):
			start = i*batch_size
			end = min((i+1)*batch_size,25000)
			x = data_train[start:end]
			y = label_train[start:end]

			sess.run(train_step, feed_dict={xs:x, ys:y})

		x_t = data_test[epoch*batch_size:(epoch+1)*batch_size]
		y_t = label_test[epoch*batch_size:(epoch+1)*batch_size]

		result = sess.run(merged, feed_dict={xs:x, ys:y})
		train_writer.add_summary(result, epoch)

		result = sess.run(merged, feed_dict={xs:x_t, ys:y_t})
		validate_writer.add_summary(result, epoch)

		pred = sess.run(prediction, feed_dict={xs:x_t, ys:y_t})

		print('epoch:', epoch ,'accuracy:', sess.run(accuracy, feed_dict={xs:x_t, ys:y_t}), 'AUC:', roc_auc_score(y_t, pred), 'AUPRC:', average_precision_score(y_t, pred))

		if (sess.run(accuracy, feed_dict={xs:x_t, ys:y_t}) >= 0.88):
			break

	fpr, tpr, threshold = roc_curve(y_t, pred)
	roc_auc = auc(fpr,tpr)

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
	         lw=lw, label='AUC = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic(RNN, test)')
	plt.legend(loc="lower right")
	plt.show()

	precision, recall, _ = precision_recall_curve(y_t, pred)

	plt.plot(recall, precision, color='darkorange',
	         lw=lw, label='AUPRC %0.2f)' % average_precision_score(y_t, pred))
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Precision')
	plt.ylabel('Recall')
	plt.title('Precision-recall curve(RNN, test)')
	plt.legend(loc="lower right")
	plt.show()
