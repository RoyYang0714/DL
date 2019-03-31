import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def p1 (X, Y, X_t):
	w = np.zeros((56,1))
	bias = np.zeros((800, 1))
	wt = w
	bt = bias

	MSE = np.mean((np.dot(X, w)+bias-Y)**2) + np.mean(np.dot(np.transpose(w), w))/2

	for i in range(800):
		x = X[i].reshape((56,1))
		y = Y[i].reshape((1, 1))
		b = bias[i].reshape((1,1))

		w_deriv = w + w*x**2/400 - x*y/400 - x*b/400
		b_deriv = np.dot(X, w)/400 + bias/400 - Y/400
		wt -= w_deriv * 0.02
		bt -= b_deriv * 0.02

		if MSE > np.mean((np.dot(X, wt)+bt-Y)**2) + np.mean(np.dot(np.transpose(w), w))/2:
			w = wt
			b = bt
			MSE = np.mean((np.dot(X, w)+bias-Y)**2) + np.mean(np.dot(np.transpose(w), w))/2
		else:
			w = w
			bias = bias

	w = w[0:54]
	i = np.arange(1001,1045,1)
	i = np.transpose(i)
	i = i.reshape((44,1))
	p = np.dot(X_t, w)

	p = np.hstack((i,p))

	df = pd.DataFrame({'Column1':p[:,0],'Column2':p[:,1]})
	np.savetxt(r"105061129_1.txt", df.values, fmt='%d	%1.1f')	

def p2 (X, Y, X_t):
	w = np.zeros((56,1))	
	bias = np.zeros((800, 1))
	wt = w
	bt = bias

	for i in range(1000):
		pt = sigmoid(np.dot(X, w))
		gradient = np.dot(np.transpose(X), pt-Y)/800
		w -= gradient * 0.01

	w = w[0:54]
	p = np.dot(X_t, w)

	i = np.arange(1001,1045,1)
	i = np.transpose(i)
	i = i.reshape((44,1))

	p = np.hstack((i,p))

	df = pd.DataFrame({'Column1':p[:,0],'Column2':p[:,1]})
	np.savetxt(r"105061129_2.txt", df.values, fmt='%d	%1.1f')