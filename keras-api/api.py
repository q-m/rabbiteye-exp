#!/usr/bin/env python3
import os
import json
from flask import Flask, jsonify, render_template, request
from lib.featurizer import Featurizer
from lib.categorizer import Categorizer
from lib.classifier import Classifier

app = Flask(__name__)

featurizer = Featurizer.load('model/reuters')
model = Classifier.load('model/reuters')
categorizer = Categorizer.load('model/reuters_y')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/v1/predict', methods=['POST', 'GET']) # GET for easy debugging
def predict():
    data = request.get_json() or request.args
    x = featurizer.transform([data])
    y = model.predict(x)
    r = categorizer.transform_inv(y)
    return jsonify({ 'result': int(r[0]) })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port)
