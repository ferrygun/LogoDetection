import os
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet import models
import numpy as np
import argparse
import cv2

from flask import Flask, request, render_template, json
from skimage import io
import urllib.request

import tensorflow as tf
graph = tf.get_default_graph()

# load the class label mappings
LABELS = open("retinanet_classes.csv").read().strip().split("\n")
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

# load the model from disk
model = models.load_model("output.h5", backbone_name="resnet50")

port = int(os.getenv("PORT"))
#port = 3000
confidence = 0.5

def processImage(url):
  response = urllib.request.urlopen(url)
  image = np.asarray(bytearray(response.read()), dtype=np.uint8)
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)

  output = image.copy()
  image = preprocess_image(image)
  (image, scale) = resize_image(image)
  image = np.expand_dims(image, axis=0)

  # detect objects in the input image and correct for the image scale
  (boxes, scores, labels) = model.predict_on_batch(image)
  boxes /= scale

  coords = []

  # loop over the detections
  for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
	# filter out weak detections
    if score < confidence:
	    continue

	# convert the bounding box coordinates from floats to integers
    box = box.astype("int")

	# build the label and draw the label + bounding box on the output
	# image
    label = "{}: {:.2f}".format(LABELS[label], score)
    coord = (label, str(box[0]), str(box[1]), str(box[2]), str(box[3]))
    coords.append(coord)

  return coords


### Flask Setup ###
app = Flask(__name__)
@app.route('/img', methods = ['GET'])
def index():
    url = request.query_string.decode("utf-8") 
    url = url[4:]
    print(url)

    global graph
    with graph.as_default():
        result = processImage(url)
    print(result)
    return (json.dumps(result))
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
