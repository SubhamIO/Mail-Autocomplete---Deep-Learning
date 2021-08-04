#!flask/bin/python
import os
import sys
print("===app.py ----- PATH===", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(""+os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, make_response, request, jsonify
from flask_cors import CORS
from flask_restful import Resource, Api

from resources.mail_sentence_completer import mail_sentence_completer


import config
from simplexml import dumps
import json
import config
#import nltk
#nltk.download()
#nltk.download('stopwords')
import logging
from logging.handlers import RotatingFileHandler
from time import strftime
import traceback

def output_xml(data, code, headers=None):
    """Makes a Flask response with a XML encoded body"""
    resp = make_response(dumps({'response' :data}), code)
    resp.headers.extend(headers or {})
    return resp



# creating the flask app
app = Flask(__name__)
CORS(app)


# creating an API object
api = Api(app, default_mediatype='application/json')
api.representations['application/xml'] = output_xml
# adding the defined resources along with their corresponding urls

api.add_resource(mail_sentence_completer, '/services/mail_sentence_completer')

# Load the default configuration
app.config.from_object(config)



# driver function
if __name__ == '__main__':
    app.run(port = 5001,debug = True)
