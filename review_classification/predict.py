import os
import json

import pandas as pd
import numpy as np

from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from review_classification.preprocessors import clean_text, get_sequences

# name for user text input
TEXTCOLUMN = 'review'

# get data paths
dirname = os.path.dirname(__file__)
tokenizer_path = os.path.join(dirname, 'data/tokenizer.json')
model_path = os.path.join(dirname, 'data/lstm_model.h5')

# load saved tokenizer and saved model
with open(tokenizer_path, 'r') as f:
    tokenizer_json = json.load(f)

tokenizer = tokenizer_from_json(tokenizer_json)

lstm_model = load_model(model_path)


def make_prediction_json(input_data):
    """Making a prediction using the preprocessing functions,
    the saved tokenizer and the saved model.
    input_data is a json-string."""

    # convert data to pandas data frame
    data = pd.read_json(input_data)[TEXTCOLUMN]

    # preprocess data
    cleaned_data = data.apply(lambda x: clean_text(x))
    model_input = get_sequences(tokenizer, cleaned_data)

    # make predictions
    predictions = lstm_model.predict(model_input)
    predictions = np.argmax(predictions, axis=1)

    # return response for api
    response = {'predictions': predictions}

    return response


def make_prediction_raw(input_data):
    """Making a prediction using the preprocessing functions,
        the saved tokenizer and the saved model.
        input_data is an array with strings for different reviews."""

    # preprocess data
    cleaned_data = [clean_text(x) for x in input_data]
    model_input = get_sequences(tokenizer, cleaned_data)

    # make predictions
    predictions = lstm_model.predict(model_input)
    predictions = np.argmax(predictions, axis=1)

    return predictions
