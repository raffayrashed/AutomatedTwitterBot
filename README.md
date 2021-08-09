Trump Twitter Bot - CS 175 W21 Project 
Created by: Lenah Syed, Raffay Rashed, Ramsey Shafi


main.py: entry point to our program, runs all of our files and generates a tweet
text_preprocessing.py: This module contains functions to load the datasets, pre-process and tokenize the data, and
perform sentiment analysis on the dataset
word2vec.py: builds and train Word2Vec model and also can save and load it to a file
sequences.py: generates sequences and encodes them; to be used in our RNN
rnn.py: builds and trains an LSTM RNN model, and also generates text from that model
sentiment_dict.py: loads in the NRC Emotion Lexicon and generates a sentiment dict by extracting only the info we need for sentiment analysis
twitter_secrets.py: contains constant variables with information pertaining to a Twitter developer account
twitter_api_module.py: posts a tweet to a Twitter developer account


If you want to post your tweets to a Twitter account, you will need a twitter developer account and populate the appropriate access tokens
into a twitter_secrets.py module (we did not push this file to our git repository)

Directories/Files that are ignored by our .gitignore:
__pycache__
models
wordvectors
NRC-Emotion-Lexicon
twitter_secrets.py
__pycache__
models
wordvectors
NRC-Emotion-Lexicon
twitter_secrets.py
trump_tweets.json
sentiment_dict.txt
trump_tweets_sentiment.txt

Our application makes use of the NRC Emotion Lexicon created by Saif M. Mohammad and Peter D. Turney at the National Research Council Canada. http://saifmohammad.com/WebPages/lexicons.html

Our application also utilizes the Every Donald Trump Tweet Dataset found on the Trump Archive Twitter: https://www.thetrumparchive.com/


