import tensorflow as tf
import numpy as np
from collections import Counter
import os
import pickle
import copy
import matplotlib.pyplot as plt

def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    return data

def create_lookup_tables(text):
    CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }

    vocab = set(text.split())
    vocab_to_int = copy.copy(CODES)

    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i

    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):

    source_text_id = []
    target_text_id = []
    
    # make a list of sentences (extraction)
    source_sentences = source_text.split("\n")
    target_sentences = target_text.split("\n")
    
    max_source_sentence_length = max([len(sentence.split(" ")) for sentence in source_sentences])
    max_target_sentence_length = max([len(sentence.split(" ")) for sentence in target_sentences])
    
    # iterating through each sentences (# of sentences in source&target is the same)
    for i in range(len(source_sentences)):
        # extract sentences one by one
        source_sentence = source_sentences[i]
        target_sentence = target_sentences[i]
        
        # make a list of tokens/words (extraction) from the chosen sentence
        source_tokens = source_sentence.split(" ")
        target_tokens = target_sentence.split(" ")
        
        # empty list of converted words to index in the chosen sentence
        source_token_id = []
        target_token_id = []
        
        for index, token in enumerate(source_tokens):
            if (token != ""):
                source_token_id.append(source_vocab_to_int[token])
        
        for index, token in enumerate(target_tokens):
            if (token != ""):
                target_token_id.append(target_vocab_to_int[token])
                
        # put <EOS> token at the end of the chosen target sentence
        # this token suggests when to stop creating a sequence
        target_token_id.append(target_vocab_to_int['<EOS>'])
            
        # add each converted sentences in the final list
        source_text_id.append(source_token_id)
        target_text_id.append(target_token_id)
    
    return source_text_id, target_text_id

def enc_dec_model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets') 
    
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)    
    
    return inputs, targets, target_sequence_length, max_target_len

def hyperparam_inputs():
    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return lr_rate, keep_prob

def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    # get '<GO>' id
    go_id = target_vocab_to_int['<GO>']
    
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
    
    return after_concat

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, source_vocab_size, encoding_embedding_size):

    embed = tf.contrib.layers.embed_sequence(rnn_inputs, vocab_size=source_vocab_size, embed_dim=encoding_embedding_size)
    
    stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])
    
    outputs, state = tf.nn.dynamic_rnn(stacked_cells, embed, dtype=tf.float32)

    return outputs, state

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_summary_length, output_layer, keep_prob):

    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
    
    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_summary_length)
    
    return outputs


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob):

    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, tf.fill([batch_size], start_of_sequence_id), end_of_sequence_id)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer)
    
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)

    return outputs

def decoding_layer(dec_input, encoder_state, target_sequence_length, max_target_sequence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):

    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state, cells, dec_embed_input, target_sequence_length, max_target_sequence_length, output_layer, keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state, cells, dec_embeddings, target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'], 
        									max_target_sequence_length, target_vocab_size, output_layer,batch_size,keep_prob)

    return (train_output, infer_output)

def seq2seq_model(input_data, target_data, keep_prob, batch_size, target_sequence_length, max_target_sentence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):

    enc_outputs, enc_states = encoding_layer(input_data, rnn_size, num_layers, keep_prob, source_vocab_size, enc_embedding_size)
    
    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)
    
    train_output, infer_output = decoding_layer(dec_input, enc_states, target_sequence_length, max_target_sentence_length, rnn_size, num_layers,
                                              target_vocab_to_int, target_vocab_size, batch_size, keep_prob, dec_embedding_size)
    
    return train_output, infer_output

def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

def save_params(params):
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)

def load_params():
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)

### hyperparameters ###
display_step = 300

epochs = 10
batch_size = 128

rnn_size = 128
num_layers = 3

encoding_embedding_size = 200
decoding_embedding_size = 200

learning_rate = 0.001
keep_probability = 0.5

### data loading ###    
source_text = load_data('en.txt')
target_text = load_data('fr.txt')

### data prpeocessing ###
source_text = source_text.lower()
target_text = target_text.lower()

# create lookup tables
source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

# create list of sentences whose words are represented in index
source_int_text, target_int_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, target_sequence_length, max_target_sequence_length = enc_dec_model_inputs()
    lr, keep_prob = hyperparam_inputs()
    
    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, target_sequence_length, max_target_sequence_length,
                                                   len(source_vocab_to_int), len(target_vocab_to_int), encoding_embedding_size, decoding_embedding_size,
                                                   rnn_size, num_layers, target_vocab_to_int)
    
    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    # - Returns a mask tensor representing the first N positions of each cell.
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function - weighted softmax cross entropy
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]

valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths = next(get_batches(valid_source, valid_target, batch_size, source_vocab_to_int['<PAD>'], target_vocab_to_int['<PAD>']))                                                                                                  

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    tac = []
    vac = []

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 keep_prob: keep_probability})


            if batch_i % display_step == 0 and batch_i > 0:
                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})

                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)
                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))
        
        tac.append(train_acc)
        vac.append(valid_acc)

	# Save Model
    saver = tf.train.Saver()
    saver.save(sess, 'checkpoints/dev')
    print('Model Trained and Saved')
    save_params('checkpoints/dev')

def sentence_to_seq(sentence, vocab_to_int):
    results = []
    for word in sentence.split(" "):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])
            
    return results
   
source_text = load_data('en.txt')
source_text = source_text.lower()

source_sentences = source_text.split("\n")

translate_sentence = source_sentences[0]

translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

load_path = load_params()
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size, target_sequence_length: [len(translate_sentence)*2]*batch_size, keep_prob: 1.0})[0]

print('Source(English)')
print('  Word Indices:  {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nTranslation(French)')
print('  Word Indices: {}'.format([i for i in translate_logits]))
print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits]))) 

### plot learning curve ###
plt.plot(tac, '--', color="#111111",  label="Training accuracy")
plt.plot(vac, color="#111111", label="Validation accuracy")

# Create plot
plt.title("Learning Curve")
plt.xlabel("epochs"), plt.ylabel("Accuracy"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

source_text = load_data('test.txt')
source_text = source_text.lower()

source_sentences = source_text.split("\n")

tmp = []

load_path = load_params()
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
	
	# Load saved model
	loader = tf.train.import_meta_graph(load_path + '.meta')
	loader.restore(sess, load_path)

	input_data = loaded_graph.get_tensor_by_name('input:0')
	logits = loaded_graph.get_tensor_by_name('predictions:0')
	target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
	keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

	for i in range(len(source_sentences)-1):
		
		translate_sentence = source_sentences[i]

		translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

		translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size, target_sequence_length: [len(translate_sentence)*2]*batch_size, keep_prob: 1.0})[0]

		tmp.append('{}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))

	tmp = np.asarray(tmp)

	np.savetxt('test_105061129.txt', tmp, fmt="%s")