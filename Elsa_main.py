import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow as tf
from tensorflow import keras
import random
import json

with open("/GitHub/Elsa_Chatbot/intents.json") as file:
    data = json.load(file)

##print(data["intents"])

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        
