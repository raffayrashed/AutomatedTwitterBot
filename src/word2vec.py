# Python module that builds and train Word2Vec model and also can save and load it to a file

import numpy as np 
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from src.text_preprocessing import pre_process_text, load_tweets_from_dataset
from gensim.models.phrases import Phrases, Phraser

# generate word2vec model
def generate_weights_for_model(tokens, filename): 
    word2Vec = Word2Vec(min_count=1, negative=10, size=100, window=2, iter=100)
    newTokens = tokens
    word2Vec.build_vocab(newTokens, progress_per=10000)
    word2Vec.train(newTokens, total_examples=word2Vec.corpus_count, epochs=100, report_delay=1)
    word_vectors = word2Vec.wv
    word_vectors.save(filename)

def load_word_vecs_for_model(filename):
    word_vectors = KeyedVectors.load(filename)
    return word_vectors


