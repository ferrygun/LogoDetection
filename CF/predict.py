# USAGE
# python predict.py --model output.h5 --labels logos/retinanet_classes.csv \
# 	--image logos/images/786414.jpg --confidence 0.5

# import the necessary packages
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet import models
import numpy as np
import argparse
import cv2

from skimage import io
import urllib.request


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
ap.add_argument("-l", "--labels", required=True,
	help="path to class labels")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load the class label mappings
LABELS = open(args["labels"]).read().strip().split("\n")
print(LABELS)
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}
print(LABELS)

# load the model from disk
model = models.load_model(args["model"], backbone_name="resnet50")

# load the input image (in BGR order), clone it, and preprocess it
#image = read_image_bgr(args["image"])
#print(image)

print("--------------------")
response = urllib.request.urlopen("http://nationalpainreport.com/wp-content/uploads/2014/12/Lyrica-75-apteka-Erudita-480x350.jpg")
image = np.asarray(bytearray(response.read()), dtype=np.uint8)
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#image = cv2.resize(image, (224, 224), inhttps://www.mims.com/resources/drugs/Philippines/packshot/Norvasc6001PPS0.JPGterpolation=cv2.INTER_CUBIC)

#print(image)

output = image.copy()
image = preprocess_image(image)
(image, scale) = resize_image(image)
image = np.expand_dims(image, axis=0)

# detect objects in the input image and correct for the image scale
(boxes, scores, labels) = model.predict_on_batch(image)
boxes /= scale

# loop over the detections
for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
	# filter out weak detections
	if score < args["confidence"]:
		continue

	# convert the bounding box coordinates from floats to integers
	box = box.astype("int")

	# build the label and draw the label + bounding box on the output
	# image
	label = "{}: {:.2f}".format(LABELS[label], score)
	print(label)
	cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
		(0, 255, 0), 2)
	print(box[0])
	print(box[1])
	print(box[2])
	print(box[3])
	cv2.putText(output, label, (box[0], box[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
