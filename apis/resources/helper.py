import warnings
warnings.filterwarnings("ignore")

import unicodedata
import pandas as pd
import re
import time
import warnings
import numpy as np
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from collections import Counter
from scipy.sparse import hstack
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
from flask import jsonify, request
from flask_restful import Resource
from flask import current_app
import json
import pickle
import random, string
import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, Dropout, LSTMCell, RNN, Bidirectional, Concatenate, Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K
from IPython.display import Image

def model_loader(load_path):
    enc_model = keras.models.load_model(load_path+'/encoder-model-final.h5', compile=False)
    inf_model = keras.models.load_model(load_path+'/inf-model-final.h5', compile=False)
    return enc_model , inf_model

def tokenize_text(text,tokenizer,max_length_in,max_length_out):
  text = '<start> ' + text.lower() + ' <end>'
  text_tensor = tokenizer.texts_to_sequences([text])
  text_tensor = keras.preprocessing.sequence.pad_sequences(text_tensor, maxlen=max_length_in, padding="post")
  return text_tensor


# Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
def decode_sequence(input_sentence,enc_model ,inf_model,tokenizer):

    max_length_in = 21
    max_length_out = 20

    sentence_tensor = tokenize_text(input_sentence,tokenizer,max_length_in,max_length_out)
    # Encode the input as state vectors.
    state = enc_model.predict(sentence_tensor)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']
    curr_word = "<start>"
    decoded_sentence = ''

    i = 0
    while curr_word != "<end>" and i < (max_length_out - 1):
        print(target_seq.shape)
        output_tokens, h = inf_model.predict([target_seq, state])

        curr_token = np.argmax(output_tokens[0, 0])

        if (curr_token == 0):
          break;

        # Reversed map from a tokenizer index to a word
        index_to_word = dict(map(reversed, tokenizer.word_index.items()))

        
        curr_word = index_to_word[curr_token]

        decoded_sentence += ' ' + curr_word
        target_seq[0, 0] = curr_token
        state = h
        i += 1

    return decoded_sentence
