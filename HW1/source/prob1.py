import numpy as np

def lr_nb_ps (X, Y, X_t, Y_t):
	X_plus = np.linalg.pinv(X)
	w = np.dot(X_plus, Y)
	p = np.dot(X_t, w)
	RMSE = np.sqrt(np.mean((p-Y_t)**2))
	return RMSE, p

def lr_nb (X, Y, X_t, Y_t):
	w = np.zeros((56,1))
	wt = w
	MSE = np.mean((np.dot(X, w)-Y)**2) + np.mean(np.dot(np.transpose(w), w))/2

	for i in range(len(X)):
		x = X[i].reshape((56,1))
		y = Y[i].reshape((1, 1))

		w_deriv = w*x**2/400 - x*y/400 + w
		wt -= w_deriv * 0.02

		if MSE > np.mean((np.dot(X, wt)-Y)**2) + np.mean(np.dot(np.transpose(w), w))/2:
			w = wt
			MSE = np.mean((np.dot(X, w)-Y)**2) + np.mean(np.dot(np.transpose(w), w))/2
		else:
			w = w

	p = np.dot(X_t, w)
	RMSE = np.sqrt(np.mean((p-Y_t)**2))

	return RMSE, p

def lr (X, Y, X_t, Y_t):
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

	RMSE = np.sqrt(np.mean((p-Y_t)**2))

	return RMSE, p

def lr_bay (X, Y, X_t, Y_t):
	w = np.zeros((56, 1))

	for i in range(1000):
		lamda_m = np.linalg.inv(np.dot(np.transpose(X), X) + np.linalg.inv(np.identity(56)))
		mu = np.dot(lamda_m, np.dot(np.transpose(X), Y))
		w = np.exp(-1/2*(np.transpose(np.dot(np.transpose(w-mu), lamda_m)) * (w-mu))) * 0.2

	p = np.dot(X_t, w)

	RMSE = np.sqrt(np.mean((p-Y_t)**2))

	return RMSE, p