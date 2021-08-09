# Python module that builds and trains an LSTM RNN model, and also generates text from that model

import tensorflow as tf
from collections import defaultdict
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from src.word2vec import load_word_vecs_for_model
import ast
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import load
import nltk

def build_rnn_model(epochs, batch_size, is_using_word2vec, sequences, num_words_size, seq_len, word2vec_filename, model_file_name):
    # load the sequences
    inputX, outputY = np.array(sequences[:, :-1]), np.array(sequences[:, -1])
    outputY = tf.keras.utils.to_categorical(outputY, num_classes=num_words_size) # one hot encodes the vectors

    # load the word2vec vectors
    if is_using_word2vec:
        word_vectors = load_word_vecs_for_model(word2vec_filename)
        weights = word_vectors.vectors
        _, embed_size = weights.shape
        new_weights = np.copy(weights)
        new_weights.resize((num_words_size, embed_size)) # resize the shape so that it is equal to len(tokenizer.word_index) + 1

    # build the model
    model = tf.keras.Sequential() # Sequential groups a linear stack of layers into a tf.keras.Model.
    if is_using_word2vec:
        model.add(tf.keras.layers.Embedding(num_words_size, 100, input_length=seq_len-1, weights=[new_weights]))
    else:
        model.add(tf.keras.layers.Embedding(num_words_size, 100, input_length=seq_len-1))
    model.add(Bidirectional(LSTM(100, return_sequences = True)))  
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(100, return_sequences = True)))
    model.add(Dense(100))
    model.add(LSTM(100))
    model.add(Dense(200))  
    model.add(Activation('relu'))
    model.add(Dense(num_words_size)) 
    model.add(Activation('softmax'))

    print(model.summary())

    # compile and train the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(inputX, outputY, batch_size=batch_size, epochs=epochs)
   
    model.save(model_file_name)

def load_model_for_rnn(model_filename):
    model = load_model(model_filename)
    return model

# generate text using the model
def generate_text(seed_text, word2vec_filename, model):
    word_vectors = load_word_vecs_for_model(word2vec_filename)
    result_text = seed_text.capitalize()
    count = 0
    last_word = result_text.split(' ')[-1]
    last_word_pos = nltk.pos_tag([last_word], tagset="universal")[0][1]
    prev_pos = ""
    # generate at least 16 words for each tweet and make sure that it ends on a noun or stop if it generates 30 words (to prevent infinite looping)
    while (count < 15 or last_word_pos != 'NOUN') and (count < 30):
        prev_pos = last_word_pos 
        encoded_seed_list = [word_vectors.vocab[word.lower().replace('.', '')].index for word in result_text.split(' ')]
        encoded_seed_list = pad_sequences([encoded_seed_list], padding='pre')
        # predict the word using the model and seed text
        predicted_index = model.predict_classes(encoded_seed_list, verbose=0)[0] 
        predicted_word = word_vectors.index2word[predicted_index]
        # capitalize 'I'
        if (predicted_word == 'i'):
            predicted_word = 'I'
        result_text += " " + predicted_word
        count += 1
        last_word = result_text.split(' ')[-1]
        last_word_pos = nltk.pos_tag([last_word], tagset="universal")[0][1]
        # puncutate the tweet using part of speech tagging
        if last_word_pos == 'PRON' and prev_pos == 'NOUN':
            result_text_tokens = result_text.split(' ')
            result_text = ""
            for i in range(len(result_text_tokens)):
                if i != 0:
                    result_text += ' '
                if i == len(result_text_tokens) - 2:
                    result_text += result_text_tokens[i] +  "."
                elif i == len(result_text_tokens) - 1:
                    result_text += result_text_tokens[i].capitalize()
                else:
                    result_text += result_text_tokens[i]
    # always end the tweet on an exclamation mark
    result_text += "!"
    result_text = result_text.replace("_", " ")
    return result_text
            