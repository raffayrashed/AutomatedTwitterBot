# Python module that loads in the NRC Emotion Lexicon and generates a sentiment dict by extracting only the info we need for sentiment analysis
# We wrote this dict to a file: sentiment_dict.txt

import json
from collections import defaultdict
import ast

# generates the sentiment dict to be used in sentiment analysis (makes use of the sentiment lexicon)
def get_sentiment_dict(): 
    infile = open('lexicon.json',encoding='utf-8')
    lexicon_json = json.load(infile, encoding="utf-8")
    result = defaultdict(dict)
    for d in lexicon_json:
        result[d["English (en)"]]["Positive"] = d["Positive"]
        result[d["English (en)"]]["Negative"] = d["Negative"]
        result[d["English (en)"]]["Anger"] = d["Anger"]
        result[d["English (en)"]]["Anticipation"] = d["Anticipation"]
        result[d["English (en)"]]["Disgust"] = d["Disgust"]
        result[d["English (en)"]]["Fear"] = d["Fear"]
        result[d["English (en)"]]["Joy"] = d["Joy"]
        result[d["English (en)"]]["Sadness"] = d["Sadness"]
        result[d["English (en)"]]["Surprise"] = d["Surprise"]
        result[d["English (en)"]]["Trust"] = d["Trust"]
    infile.close()
    return result 
