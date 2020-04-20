# import 
import numpy as np
import argparse
import cv2

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to the input b&w image")
ap.add_argument("-p", "--prototext", type=str, required=True,
	help="path to Caffee prototext file")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to Caffee pre-trained model")
ap.add_argument("-c", "--points", type=str, required=True,
	help="path to cluster center points")
args = vars(ap.parse_args())

# load model and cluster points
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototext"], 
								args["model"])
pts = np.load(args["points"])

# add the cluster centers as 1x1 convolutions 
# to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606,
								dtype="float32")]

# load image
image = cv2.imread(args["image"])
# scale image to pixel intensities between 0 & 1
scaled = image.astype("float32") / 255.0
# convert BGR image to Lab color space
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# resize the image to 224x244 dimension
resized = cv2.resize(lab, (224, 224))
# split the channels and extract the L channel
# Lab color space has 3 chanels
# L: light, a: green-red, b: blue-yellow
L = cv2.split(resized)[0]
# mean-centering
L -= 50

# pass the L channel image to the network 
# which will predict the a and b channel values
print("[INFO] colorizing image...")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# resiae the predicted ab values to the original 
# image dimension
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# concatenate the ab channel with the L channel
# of the original input image
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab),
							axis=2)

# convert the lab color image to BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
# clip any values outside the range [0, 1]
colorized = np.clip(colorized, 0, 1)

# convert the floating points in the range [0, 1]
# to integers in the range [0, 255] 
colorized = (255 * colorized).astype("uint8")

# show the original image and the colorized output
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)