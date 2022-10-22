from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

args = {
        #"dataset":"F:/crop detection/crop/dataset/train",
        "model":"E:/crop detection/crop dis/output/crop.model",
        "labelbin":"E:/crop detection/crop dis/output/category_lb.pickle",
        #"colorbin":"F:/laptop/laptop kares/multi-output-classification/output/color_lb.pickle",
        "image":"C:/Users/Hamza Javed/Desktop/New folder/l (235).JPG"

        }

image = cv2.imread(args["image"])
output = imutils.resize(image, width=400)
 
# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())

# labels with the *largest* probability
print("[INFO] classifying image...")
(categoryProba) = model.predict(image)


categoryIdx = categoryProba[0].argmax()
categoryLabel = mlb.classes_[categoryIdx]

categoryText = "category: {} ({:.2f}%)".format(categoryLabel,
    categoryProba[0][categoryIdx] * 100)
cv2.putText(output, categoryText, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 255, 0), 2)

print("[INFO] {}".format(categoryText))

cv2.imshow("Output", output)
cv2.waitKey(0)
