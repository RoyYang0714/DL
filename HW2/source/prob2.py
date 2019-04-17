import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lr (X, Y, X_t, Y_t, th):
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

	p = np.dot(X_t, w)

	p[p > th]=1
	p[p <= th]=0

	return p

def lgr (X, Y, X_t, Y_t, th):
	w = np.zeros((56,1))
	bias = np.zeros((800, 1))
	wt = w
	bt = bias

	for i in range(1000):
		pt = sigmoid(np.dot(X, w))
		gradient = np.dot(np.transpose(X), pt-Y)/800
		w -= gradient * 0.01

	p = np.dot(X_t, w)

	p[p > th]=1
	p[p <= th]=0

	return p

def con_mat (p, Y_t):
	con = pd.DataFrame(np.zeros((2,2)), columns=['true0','true1'])
	
	for i in range(200):
		if Y_t[i] == 0 and p[i] == 0:
			con.iloc[0, 0]+=1
		elif Y_t[i] == 0 and p[i] == 1:
			con.iloc[0, 1]+=1
		elif Y_t[i] == 1 and p[i] == 0:
			con.iloc[1, 0]+=1
		else:
			con.iloc[1, 1]+=1

	return con