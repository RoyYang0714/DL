import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot(data):
	tsne = TSNE(n_components=2)
	X_tsne = tsne.fit_transform(data)

	x = X_tsne[:, 0].reshape((1102, 1))
	y = X_tsne[:, 1].reshape((1102, 1))

	cate = data.loc[:, ['Activities_Types']]
	cate = cate.values.reshape((1102, 1))

	x = np.c_[x, cate]

	x_t = np.zeros((1, 1))
	y_t = np.zeros((1, 1))

	label = ['0', 'dws', 'ups', 'sit', 'std', 'wlk', 'jog']
	colors = ['black', 'blue', 'purple', 'yellow', 'g', 'red', 'aqua']
	for i in range(1,7):
		flag = 0
		for j in range(1102):
			if x[j, 1] == i:
				if flag != 0:
					tmp = x[j, 0]
					x_t = np.r_[x_t, tmp]
					tmp = y[j, 0]
					y_t = np.r_[y_t, tmp]
				else:
					x_t = x[j, 0]
					y_t = y[j, 0]
					flag = 1	
		plt.scatter(x_t, y_t, c=colors[i], label=label[i])
	plt.legend(loc='upper right')
	plt.title("t-SNE")
	plt.show()