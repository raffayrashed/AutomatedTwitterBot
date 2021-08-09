# Module that builds a model and generates text utilziing the methods from our other python modules located in this project folder
# Entry point to our project

import src.text_preprocessing as text_preprocessing
import src.sequences as sequences
import src.word2vec as word2vec
import src.rnn as rnn
from gensim.models import KeyedVectors
from pathlib import Path
import os
from src.twitter_api_module import post_tweet
import tensorflow as tf

# Model Parammeters
EPOCHS = 100
BATCH_SIZE = 50
IS_USING_WORD2VEC = True
IS_USING_FULL_DATASET = True
RATIO = 1
ADDITIONAL_NOTES = ""


# build rnn using module functions 
def build_model_for_sentiment(sentiment):
     WORD2VEC_FILENAME = "src/wordvectors/e" + str(EPOCHS) + "bs" + str(BATCH_SIZE) + ("wv" if IS_USING_WORD2VEC else "") \
     + ("ratio" + str(RATIO) if not IS_USING_FULL_DATASET else "") + sentiment + ADDITIONAL_NOTES + "word_vectors.kv"
     MODEL_FILENAME = "src/models/e" + str(EPOCHS) + "bs" + str(BATCH_SIZE) + ("wv" if IS_USING_WORD2VEC else "") \
     + ("ratio" + str(RATIO) if not IS_USING_FULL_DATASET else "") + sentiment + ADDITIONAL_NOTES + "model.h5"
     TOKENS_FILENAME = "src/tokens/e" + str(EPOCHS) + "bs" + str(BATCH_SIZE) + ("wv" if IS_USING_WORD2VEC else "") \
     + ("ratio" + str(RATIO) if not IS_USING_FULL_DATASET else "") + sentiment + ADDITIONAL_NOTES + "tokens.txt"
     data = text_preprocessing.load_tweets_with_sentiment()
     data = text_preprocessing.get_sentiment_data(sentiment, data)
     tokens = text_preprocessing.pre_process_text(data, fullDataset=IS_USING_FULL_DATASET, ratio=RATIO)
     text_preprocessing.write_tokens_to_file(tokens, TOKENS_FILENAME)
     word2vec.generate_weights_for_model(tokens, WORD2VEC_FILENAME)
     encoded_sequences, num_words_from_sequences, max_sequence_len = sequences.generate_word_sequences(tokens, WORD2VEC_FILENAME)

     rnn.build_rnn_model(EPOCHS, BATCH_SIZE, IS_USING_WORD2VEC, encoded_sequences, num_words_from_sequences, max_sequence_len, WORD2VEC_FILENAME, MODEL_FILENAME)

# generate sentiment text using module functions
def generate_text_for_sentiment(sentiment):
     WORD2VEC_FILENAME = "src/wordvectors/e" + str(EPOCHS) + "bs" + str(BATCH_SIZE) + ("wv" if IS_USING_WORD2VEC else "") \
     + ("ratio" + str(RATIO) if not IS_USING_FULL_DATASET else "") + sentiment + ADDITIONAL_NOTES + "word_vectors.kv"
     MODEL_FILENAME = "src/models/e" + str(EPOCHS) + "bs" + str(BATCH_SIZE) + ("wv" if IS_USING_WORD2VEC else "") \
     + ("ratio" + str(RATIO) if not IS_USING_FULL_DATASET else "") + sentiment + ADDITIONAL_NOTES + "model.h5"
     TOKENS_FILENAME = "src/tokens/e" + str(EPOCHS) + "bs" + str(BATCH_SIZE) + ("wv" if IS_USING_WORD2VEC else "") \
     + ("ratio" + str(RATIO) if not IS_USING_FULL_DATASET else "") + sentiment + ADDITIONAL_NOTES + "tokens.txt"
     tokens = text_preprocessing.load_tokens_from_file(TOKENS_FILENAME)
     seed = text_preprocessing.get_random_seed(tokens, n=2, fullDataset=IS_USING_FULL_DATASET, ratio=RATIO)
     model  = rnn.load_model_for_rnn(MODEL_FILENAME)
     result_text = rnn.generate_text(seed, WORD2VEC_FILENAME, model)
     return result_text

if __name__ == '__main__':
     # if the models/wordvectors directories don't already exist in your directory, create them now
     if not Path('models').is_dir():
          os.makedirs('models', exist_ok=True)
     if not Path('wordvectors').is_dir():
          os.makedirs('wordvectors', exist_ok=True)
     if not Path('tokens').is_dir():
          os.makedirs('tokens', exist_ok=True)


     gpus = tf.config.list_physical_devices('GPU')
     if gpus:
          try:
               # Currently, memory growth needs to be the same across GPUs
               for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
               logical_gpus = tf.config.experimental.list_logical_devices('GPU')
               print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
          except RuntimeError as e:
               # Memory growth must be set before GPUs have been initialized
               print(e)


     # if model's haven't been built, uncomment and run the lines below:
     # build_model_for_sentiment('Positive')
     # build_model_for_sentiment('Negative')
     # build_model_for_sentiment('Anger')
     # build_model_for_sentiment('Anticipation')
     # build_model_for_sentiment('Disgust')
     # build_model_for_sentiment('Fear')
     # build_model_for_sentiment('Joy')
     # build_model_for_sentiment('Sadness')
     # build_model_for_sentiment('Surprise')
     # build_model_for_sentiment('Trust')

     # uncomment the line below if you want to prompt the user for a sentiment input
     #user_sentiment = input("Please enter a sentiment that you'd like the Trump bot to generate [Positive, Negative, Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust]: ")
     user_sentiment = 'Positive'
     result_text = generate_text_for_sentiment(user_sentiment)
     
     print(result_text) # outputs result_text to the console

     # comment out if not posting directly to twitter
     # post_tweet(result_text) # tweets result_text directly to our twitter account


