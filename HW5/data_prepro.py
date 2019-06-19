import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_prepro():
	df = pd.read_csv('data/emnist-balanced-train.csv', header=None)
	df = df.values

	data = []
	label = []

	for i in range(len(df)):
		if (df[i][0] > 9 and df[i][0] < 36):
			label.append(df[i][0])
			data.append(df[i][1:]/255)

	label = np.asarray(label).reshape((len(label), 1))
	data = np.asarray(data)

	data = data.reshape((len(data), 28, 28))
	data = np.transpose(data, (0, 2, 1))
	
	plt.figure()
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.imshow(data[i], cmap='gray')
		plt.axis('off')
	plt.show()
	
	data = data + np.random.normal(0, 0.1, data.shape)

	data = data.reshape((len(data), 784))

	return data