# Python module that generates sequences and encodes them; to be used in our RNN

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import enchant 
import numpy as np
import os
import time
import json
from random import randint
from pickle import load
import nltk
from nltk.tokenize import TweetTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from src.text_preprocessing import load_tweets_from_dataset, pre_process_text
import ast
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import dump
from gensim.models import KeyedVectors

# generates sequences of text from rnn
def generate_word_sequences(tokens_list, word2vec_filename):
    word_vectors = KeyedVectors.load(word2vec_filename)
    encoded_sequences = []
    length = 5 
    for tweet in tokens_list:
        for i in range(length, len(tweet) + 1):
            sequence = tweet[i-length:i]
            encoded_sequences.append([word_vectors.vocab[word].index for word in sequence])
    num_words_from_sequences = word_vectors.vectors.shape[0]
    max_sequence_len = max([len(x) for x in encoded_sequences]) 
    encoded_sequences = np.array(pad_sequences(encoded_sequences))
    return [encoded_sequences, num_words_from_sequences, max_sequence_len]