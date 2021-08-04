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
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, Dropout, LSTMCell, RNN, Bidirectional, Concatenate, Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K
from IPython.display import Image
from resources.helper import *
print("===mail_sentence_completer.py ----- PATH===", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class mail_sentence_completer(Resource):


    # Constructor for initializing global variable
    def __init__(self):
        self.load_path = '/Users/subham/Desktop/mail sentence completer/models'
        self.enc_model ,self.inf_model = model_loader(self.load_path)

    # Corresponds to GET request
    def get(self):
        return jsonify({'message': 'Method : GET : Working Fine!!!'})

    # Corresponds to POST request
    def post(self):
        print("\n===================== AIMLAutomation POST ==============")

        date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        jdata = jsonify({"Status":"Failure", "prediction":"Failure"})
        # Convert the request json data to json file
        if request.form.get('user_id') and request.form.get('input_sentence'):

            try:
                user_id = request.form.get('user_id')

                input_sentence = request.form.get('input_sentence')

                # datetime object containing current date and time
                now = datetime.datetime.now()
                # ddmmYYHMS
                dt_string = now.strftime("%d%m%Y%H%M%S")

                vocab_max_size = 10000

                with open(self.load_path+'/word_dict-final.json') as f:
                    word_dict = json.load(f)
                    tokenizer = keras.preprocessing.text.Tokenizer(filters='', num_words=vocab_max_size)
                    tokenizer.word_index = word_dict


                texts = [input_sentence]

                output = list(map(lambda text: (text, decode_sequence(text,self.enc_model ,self.inf_model,tokenizer)), texts))
                output_df = pd.DataFrame(output, columns=["input", "output"])
                output_df.head(len(output))

                ans = output_df['output'][0]
                ans = ans.replace('<end>','')


                jdata = jsonify({"Status":"Success", "prediction":ans})
            except Exception as ex:
                # --- Code for sending file as response ---
                print("ERRORRRRRRRRRRR Exception ---------------------------- 37")
                #print(str(ex))
                trace = []
                tb = ex.__traceback__
                while tb is not None:
                    trace.append({
                        "filename": tb.tb_frame.f_code.co_filename,
                        "name": tb.tb_frame.f_code.co_name,
                        "lineno": tb.tb_lineno
                    })
                    tb = tb.tb_next
                error_json = str({
                    'type': type(ex).__name__,
                    'message': str(ex),
                    'trace': trace})

                print(error_json)

                jdata = jsonify({"Status":"Failure", "reason":"Exception"})
        else :
            jdata = jsonify({"Status":"Failure", "reason":"Invalid Data"})

        return jdata

    def allowed_file(self, filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
