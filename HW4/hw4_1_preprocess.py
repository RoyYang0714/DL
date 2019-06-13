import tensorflow as tf

def data_load(seq_len):
	imdb = tf.keras.datasets.imdb

	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

	# A dictionary mapping words to an integer index
	word_index = imdb.get_word_index()

	# The first indices are reserved
	word_index = {k:(v+3) for k,v in word_index.items()} 
	word_index["<PAD>"] = 0
	word_index["<START>"] = 1
	word_index["<UNK>"] = 2  # unknown
	word_index["<UNUSED>"] = 3

	train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=seq_len)

	test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=seq_len)

	return train_data, test_data, train_labels, test_labels