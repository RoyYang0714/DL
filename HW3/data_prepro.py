import numpy as np
import tensorflow as tf
import pathlib
import random
import cv2

size = 128

def load_and_preprocess_image(image, path):

	image = cv2.imread(path)
	image = cv2.resize(image, (size, size), cv2.INTER_AREA)
	image = image[:,:,::-1]

	return image

def data(target):

	data_root = 'data/' + target
	data_root = pathlib.Path(data_root)

	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]

	random.shuffle(all_image_paths)

	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((name, index) for index,name in enumerate(label_names))

	all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

	num = len(all_image_labels)

	images = np.ndarray((num, size, size, 3), dtype=int)
	labels = np.zeros((num, 101))

	for i in range(num):
		images[i] = load_and_preprocess_image(images[i], all_image_paths[i])
		labels[i][all_image_labels[i]] = 1

	return images, labels, label_names