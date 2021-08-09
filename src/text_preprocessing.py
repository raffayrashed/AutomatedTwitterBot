# This module contains functions to load the datasets, pre-process and tokenize the data, and perform sentiment analysis on the dataset

import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from collections import defaultdict
import ast
import json
from nltk.corpus import stopwords
import random
import nltk
from gensim.models.phrases import Phrases, Phraser
import re

stop_words = set(stopwords.words('english')) 

# helper function for when we filer the tokenized tweets
def english(token):
    try:
        token.encode(encoding='utf-8').decode('ascii')
        return True
    except UnicodeDecodeError:
        return False

# loads the data from the trump twitter dataset and returns a json dict
def load_tweets_from_dataset():
    f = open('trump_tweets.json',encoding='utf-8')
    d = json.load(f)
    data = []
    for tweet in d:
        if tweet["isRetweet"] == "f":
            inner_dict = dict(tweet)
            data.append(inner_dict)
    return data

# returns a random seed text with n words in it (default n value is 2)
def get_random_seed(tokens, n=2, fullDataset=True, ratio=1):
    # retrieves the first n words of a random tweet
    newTokens = tokens
    if not fullDataset:
        newTokens = newTokens[:len(newTokens)//ratio]
    random_tweet = newTokens[random.randint(0, len(newTokens)) - 1]
    while len(random_tweet) < n:
        random_tweet = newTokens[random.randint(0, len(newTokens)) - 1]
    random_tweet = random_tweet[:n]
    return ' '.join(word for word in random_tweet)

# generate a list of tokenized tweets
def pre_process_text(data, fullDataset=True, ratio=1): 
    tokenizedTweets = []
    for content in data:
        list_of_sentences = nltk.sent_tokenize(content['text']) # split on punctuation
        tokenized_sentences = []
        for sent in list_of_sentences:
            word_list = sent.split(' ')
            cleaned_word_list = []
            for word in word_list: # remove punctation
                new_word = word.replace(".", "")
                new_word = new_word.replace(";", "")
                new_word = new_word.replace("!", "")
                new_word = new_word.replace("?", "")
                new_word = new_word.replace(",", "")
                new_word = new_word.replace(":", "")
                new_word = new_word.replace("-", "")
                new_word = new_word.replace(")", "")
                new_word = new_word.replace("\"", "")
                new_word = new_word.replace("(", "")
                cleaned_word_list.append(new_word)
            tokenized_sentences.extend(cleaned_word_list)
        tokenizedTweets.append(tokenized_sentences)
    # filter out non english words
    tokenizedTweets = [list(filter(lambda x: english(x), tokens)) for tokens in tokenizedTweets]
    tokenizedTweets = [list(filter(lambda x: x not in ".'?,;-!\"" and 'http' not in x and '&amp' not in x and '&' not in x, tokens)) for tokens in tokenizedTweets]
    tokenizedTweets = [list(map(lambda x: x.lower(), tokens)) for tokens in tokenizedTweets]
    tokenizedTweets = [tweet for tweet in tokenizedTweets if len(tweet) > 10]

    # bigrams
    phraseModel = Phrases(tokenizedTweets, min_count=1, threshold=2)
    bigramModel = Phraser(phraseModel)
    newTokenizedTweets = []
    for token_list in bigramModel[tokenizedTweets]:
        new_token_list = []
        for i in range(len(token_list)):
            token = token_list[i]
            new_token_list.append(token)
        newTokenizedTweets.append(new_token_list)

    tokenizedTweets = newTokenizedTweets

    if not fullDataset:
        tokenizedTweets = tokenizedTweets[:len(tokenizedTweets)//ratio]
    
    return tokenizedTweets

# modifies tweets dictionary (doesn't return anything) to assign a sentiment score for every tweet
# we pass in the sentiment_dict that was generated from sentiment_dict.py
def modify_data_with_sentiments(tweets, sentiment_dict):
    found_words = set()
    for tweet in tweets:
        pos_score = 0
        neg_score = 0
        anger_score = 0
        anticipation_score = 0
        disgust_score = 0
        fear_score = 0
        joy_score = 0
        sad_score = 0
        surprise_score = 0
        trust_score = 0
        for word in tweet["text"].split():
            if word in sentiment_dict: # TODO: might need to stem the word
                pos_score += int(sentiment_dict[word]["Positive"])
                neg_score += int(sentiment_dict[word]["Negative"])
                anger_score += int(sentiment_dict[word]["Anger"])
                anticipation_score += int(sentiment_dict[word]["Anticipation"])
                disgust_score += int(sentiment_dict[word]["Disgust"])
                fear_score += int(sentiment_dict[word]["Fear"])
                joy_score += int(sentiment_dict[word]["Joy"])
                sad_score += int(sentiment_dict[word]["Sadness"])
                surprise_score += int(sentiment_dict[word]["Surprise"])
                trust_score += int(sentiment_dict[word]["Trust"])
                found_words.add(word)
        # calculate each score by dividing the total score by the length of the tweet
        tweet["Positive"] = pos_score / len(tweet["text"])
        tweet["Negative"] = neg_score / len(tweet["text"])
        tweet["Anger"] = anger_score / len(tweet["text"])
        tweet["Anticipation"] = anticipation_score / len(tweet["text"])
        tweet["Disgust"] = disgust_score / len(tweet["text"])
        tweet["Fear"] = fear_score / len(tweet["text"])
        tweet["Joy"] = joy_score / len(tweet["text"])
        tweet["Sadness"] = sad_score / len(tweet["text"])
        tweet["Surprise"] = surprise_score / len(tweet["text"])
        tweet["Trust"] = trust_score / len(tweet["text"])

# return a list of tweets if each tweet displays at least 0.5% of the chosen sentiment
def get_sentiment_data(sentiment, data):
    result = []
    for tweet in data:
        sentiment_score = tweet[sentiment]
        if sentiment_score > 0.005: 
            result.append(tweet)
    return result

# load tweets that have sentiment scores attached to it
def load_tweets_with_sentiment():
    infile = open('trump_tweets_sentiment.txt', encoding='utf-8')
    data = eval(infile.read())
    infile.close()
    return data

# writes tokens to file to speed up text generation time
def write_tokens_to_file(tokens, filename):
    outfile = open(filename, 'w', encoding='utf-8')
    outfile.write(str(tokens))
    outfile.close()

# load tokens from file to speed up text generation time
def load_tokens_from_file(filename):
    infile = open(filename)
    tokens = eval(infile.read())
    return tokens