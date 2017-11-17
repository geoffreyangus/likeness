import numpy as np
import cv2
import re
import datetime

class FeatureExtractor():

	def extract(self,jsonBlob):
		imagePath = jsonBlob['imagePath']
		self.img = cv2.imread(imagePath)
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		self.caption = jsonBlob['caption']
		self.timeStamp = jsonBlob['timestamp']
		self.timeStampInfo()
		self.features = []
		return (np.array(self.getFeatures()),jsonBlob['likes'])

	def dominantColors(self):
		img = self.img
		arr = np.float32(img)
		pixels = arr.reshape((-1, 3))

		n_colors = 3
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		_, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

		palette = np.uint8(centroids)
		quantized = palette[labels.flatten()]
		quantized = quantized.reshape(img.shape)
		return centroids.flatten().tolist()

	def getFeatures(self):
		self.features += self.gradients()
		self.features += [self.numMentions()]
		self.features += [self.numHashtags()]
		self.features += [self.commentLength()]
		self.features += self.timeStampInfo()
		self.features += [self.numFaces()]
		self.features += [self.brightness()]
		self.features += self.dominantColors()
		return self.features

	def brightness(self):
		return np.mean(self.gray)
	# returns tuple of (dx mean, dx std, dy mean, dy std)
	def gradients(self):
		dx,dy = np.gradient(self.gray)
		return [np.mean(dx),np.std(dx),np.mean(dy),np.std(dy)]
	def numMentions(self):
		result = re.findall("@([a-zA-Z0-9]{1,15})", self.caption)
		return len(result)
	
	def commentLength(self):
		return len(self.caption)
	
	def numHashtags(self):
		result = re.findall(r"#(\w+)", self.caption)
		return len(result)

	def timeStampInfo(self):
		date = datetime.datetime.fromtimestamp(float(self.timeStamp))
		return [date.day,date.hour,float(self.timeStamp)]

	def numFaces(self):
		cascPath = "haarcascade_frontalface_default.xml"

		# Create the haar cascade
		faceCascade = cv2.CascadeClassifier(cascPath)
		# Detect faces in the image
		faces = faceCascade.detectMultiScale(
		    self.gray,
		    scaleFactor=1.1,
		    minNeighbors=5,
		    minSize=(30, 30)
		    #flags = cv2.CV_HAAR_SCALE_IMAGE
		)

		return len(faces)