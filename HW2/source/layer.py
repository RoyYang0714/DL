import tensorflow as tf

def add_layer (inputs, size_in, size_out, n_layer, activation_function = None):
	layer_name = n_layer
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			W = tf.Variable(tf.random_normal([size_in, size_out], stddev=0.1), name='W')
		with tf.name_scope('bias'):
			b = tf.Variable(tf.zeros([1, size_out]) + 0.1, name='bias')
	
	hidden_out = tf.add(tf.matmul(inputs, W), b)

	if activation_function is None:
		outputs = hidden_out
	else:
		outputs = activation_function(hidden_out)

	return outputs