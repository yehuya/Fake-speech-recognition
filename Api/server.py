#!/usr/bin/env python
# coding: utf-8
import re
import sys
import os
import numpy as np
import keras
import tensorflow as tf

from audio_process import process_wav_file
from flask import Flask, request, jsonify

from keras.models import load_model

keras.backend.clear_session()

app = Flask(__name__)
app.debug = True

# model labels
LABELS = 'bot human'.split()

# load model 
# change it to your model.h5 path
model = load_model(os.path.join(os.path.dirname(__file__), 'model.h5'))
graph = tf.get_default_graph()

def isSoundfile(str):
    aac = re.search('(.mp3)|(.wav)|(.aac)', str)

    if aac and len(aac.span()) > 1: 
        return True
    
    return False

def recognize(record):
    with graph.as_default():
        try:
            data, img = process_wav_file(record)

            # get model prediction
            prediction = model.predict(np.array([data]))

            return LABELS[np.argmax(prediction)], img
        except Exception as e: 
            print(e)
            return None, None

@app.route('/recognize', methods=['POST'])
def recognize_api():
    record = request.form.get('record')

    if record and isSoundfile(record): 
        result, img = recognize(record)

        if(result == None):
            return jsonify(
                success=False,
                message="Server error"
            )

        return jsonify(
            success=True,
            message="Success",
            data={
                "result": result,
                "spectrogram": img
            }
        )
        
    return jsonify(
        success=False,
        message="Missing audio url"
    )


if __name__ == '__main__':
    print(sys.argv[1])
    app.run('0.0.0.0', port=int(sys.argv[1]), debug=True)