import tensorflow as tf
import numpy as np
from data_prepro import *
import math
import os
import matplotlib.pyplot as plt

### data loading and preprocessing ###
data = data_prepro()

pic = data[0:4]
pic = pic.reshape(4, 28, 28)

plt.figure()
for i in range(4):
	plt.subplot(2,2,i+1)
	plt.imshow(pic[i], cmap='gray')
	plt.axis('off')
plt.show()	

### HYPERPARAMETERS ###
epochs = 100
batch_size = 32
learning_rate = 1e-3
latent_dim = 2

### VAE model ###
def weights(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def encoder(x):
	# First hidden layer
	W_fc1 = weights([784, 500])
	b_fc1 = bias([500])
	h_1   = tf.nn.softplus(tf.matmul(x, W_fc1) + b_fc1)

	# Second hidden layer 
	W_fc2 = weights([500, 501])
	b_fc2 = bias([501])
	h_2   = tf.nn.softplus(tf.matmul(h_1, W_fc2) + b_fc2)

	# Parameters for the Gaussian
	z_mean = tf.add(tf.matmul(h_2, weights([501, latent_dim])), bias([latent_dim]))
	z_log_sigma_sq = tf.add(tf.matmul(h_2, weights([501, latent_dim])), bias([latent_dim]))

	l2_loss = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)

	return z_mean, z_log_sigma_sq, l2_loss

def decoder(z):
	# First hidden layer
	W_fc1_g = weights([latent_dim, 500])
	b_fc1_g = bias([500])
	h_1_g   = tf.nn.softplus(tf.matmul(z, W_fc1_g) + b_fc1_g)

	# Second hidden layer 
	W_fc2_g = weights([500, 501])
	b_fc2_g = bias([501])
	h_2_g   = tf.nn.softplus(tf.matmul(h_1_g, W_fc2_g) + b_fc2_g)

	x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(h_2_g,  weights([501, 784])), bias([784])))

	l2_loss = tf.nn.l2_loss(W_fc1_g) + tf.nn.l2_loss(W_fc2_g)

	return x_reconstr_mean, l2_loss

x = tf.placeholder(tf.float32, [None, 784])

z_mean, z_log_sigma_sq, l2_loss_en = encoder(x)
eps = tf.random_normal((tf.shape(z_mean)[0], latent_dim), 0, 1, dtype=tf.float32)
z = z_mean + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

y, l2_loss_de = decoder(z)

l2_loss = l2_loss_de + l2_loss_en

reconstr_loss = -tf.reduce_sum(x * tf.log(1e-10 + y) + (1-x) * tf.log(1e-10 + 1 - y), 1)
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)

ELBO = -tf.reduce_mean(reconstr_loss + latent_loss)
tf.summary.scalar('lower bound', ELBO)

cost = -ELBO + 1e-4*l2_loss

optimizer =  tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("logs/", sess.graph)

    batch_iter = int(math.ceil(len(data)/batch_size))

    for epoch in range(epochs):
    	
    	for i in range(batch_iter):

    		if ((i+1)*batch_size < len(data)):
    			start = i*batch_size
    			end = (i+1)*batch_size
    			minibatch = data[start:end]

    		sess.run(optimizer, feed_dict={x:minibatch})

    	loss_tot = sess.run(cost, feed_dict={x:minibatch})
    	print("Epoch:", '%03d' % (epoch+1), 'loss', loss_tot)

    	result = sess.run(merged, feed_dict={x:minibatch})
    	train_writer.add_summary(result, epoch)

    	if epoch == 99:
    		pic_r = sess.run(y, feed_dict={x:data[0:4]})
    		pic_r = pic_r.reshape((len(pic_r), 28, 28))

    		plt.figure()
    		for i in range(4):
    			plt.subplot(2,2,i+1)
    			plt.imshow(pic_r[i], cmap='gray')
    			plt.axis('off')
    		plt.show()

    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    canvas = np.empty((28*ny, 28*nx))
    d = np.zeros([batch_size,2],dtype='float32')
    for i, yi in enumerate(x_values):
    	for j, xi in enumerate(y_values):
    		z_mu = np.array([[xi, yi]])
    		d[0] = z_mu
    		x_mean = sess.run(y, feed_dict={z: d})
    		canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper", vmin=0, vmax=1,interpolation='none',cmap=plt.get_cmap('gray'))
    plt.tight_layout()
    plt.show()

    latent = np.random.normal(0, 1, 50*latent_dim)
    latent = latent.reshape(50, latent_dim)

    pic_d = sess.run(y, feed_dict={z:latent})
    pic_d = pic_d.reshape((len(pic_d), 28, 28))

    for i in range(50):
    	plt.subplot(5, 10,i+1)
    	plt.imshow(pic_d[i], cmap='gray')
    	plt.axis('off')
    plt.show()