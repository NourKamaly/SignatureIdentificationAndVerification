from keras.models import load_model
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
import tensorflow
from tensorflow import keras
from flask import Flask, render_template, request
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import pandas as pd
import warnings
import cv2
warnings.filterwarnings("ignore")


app = Flask(__name__)

realImages={
"personA":"D:\\University Materials\\fourth year\\Computer Vision\\projects\\realImages\\personA\\A.png",
"personB":"D:\\University Materials\\fourth year\\Computer Vision\\projects\\realImages\\personB\\B.png",
"personC":"D:\\University Materials\\fourth year\\Computer Vision\\projects\\realImages\\personC\\C.png",
"personD":"D:\\University Materials\\fourth year\\Computer Vision\\projects\\realImages\\personD\\D.png",
"personE":"D:\\University Materials\\fourth year\\Computer Vision\\projects\\realImages\\personE\\E.png" 
}

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['Post'])
def predict():
    img = request.files['image']
    img_path = "./static/" + img.filename
    img.save(img_path)
    personClass = predict_person('D:\\University Materials\\fourth year\\Computer Vision\\projects\\model_VGG.h5', img_path)
    realImagePath=realImages[personClass]
    similarity=detect_similarity('D:\\University Materials\\fourth year\\Computer Vision\\projects\\model.h5', realImagePath, img_path)
    return render_template("index.html", prediction=personClass, prediction2=similarity, img_path=img_path)


def predict_person(model_classification, path):
    labels = ['personA', 'personB', 'personC', 'personD', 'personE']
    image = load_img(path, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)
    VGG_model = load_model(model_classification)
    predictions = VGG_model.predict(img)
    person = labels[predictions.argmax()]
    return person



def detect_similarity(model_similarity, real_img, img, threshold=0.5):
  image1 = cv2.imread(real_img)
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
  image1 = cv2.resize(image1, (128, 128))

  image2 = cv2.imread(img)
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
  image2 = cv2.resize(image2, (128, 128))

  im1 = preprocess_input(image1)
  im2 = preprocess_input(image2)

  model = load_model(model_similarity)
  embedding1 = model.predict(np.array([im1]))
  embedding2 = model.predict(np.array([im2]))
  distance = np.sum(np.square(embedding1-embedding2), axis=-1)
  
  L = ['Forged','Real']
  prediction = np.where(distance<=threshold, 1, 0)
  
  return L[prediction[0]]


if __name__ == '__main__':
    app.run(debug=True)
