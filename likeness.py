import numpy as np
import cv2
import util
from sklearn import linear_model

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd") # lol

class DummyFeatureExtractor():
	def extract(self, x):
		return (np.random.randint(0, 20, 15), x['likes'])


# Class: LikenessV1
# -----------------
# First iteration of the Likeness algorithm. Takes a feature vector with
# nineteen features: [numMentions, numHashtags, commentLength, numFaces, 
# imgSaturation, imgRedIntensity, imgGreenIntensity, imgBlueIntensity, 
# imgHasUserFace, dxMean, dyMean, dxStd, dyStd, date, dayOfWeek, hourOfDay, 
# avgLikes, avgCommentLength]
class LikenessV1():
	theta = np.random.random(15)
	classifier = None
	# Trains the model. Takes all data and extracts 19-vectors from the posts.
	# Runs Minibatch Gradient Descent to optimize and return a weight vector.
	def train(self, data, featureExtractor):
		X = []
		Y = []
		for example in data:
			output = featureExtractor(example)
			X.append(output[0])
			Y.append(output[1])

		X = np.asarray(X)
		Y = np.asarray(Y)

		self.classifier = linear_model.LinearRegression(fit_intercept=True, copy_X=False)
		self.classifier.fit(X, Y)
		print "parameters", self.classifier.coef_

	def predict(self, x):
		return self.classifier.predict(x)

# Class: BaseLineAlgorithm
# ------------------------
# Makes a simplistic prediction based on the assumption that the presence of a 
# face in a photo increases the number of likes on a given Instagram post.
class BaseLineAlgorithm():
	# Predicts the like count for a given user (DEPRECATED)
	def predict(self, image):
		avg = image['user']['averageLikes']
		coefficient = self.processImage(image['imagePath'])
		return avg * coefficient

	# Generates a coefficient with face recognition
	def processImage(self, imagePath):
		return 1.1 if self.detectFaces(imagePath) else 0.9

	# Source: https://realpython.com/blog/python/face-recognition-with-python/
	def detectFaces(self, imagePath):
		cascPath = "haarcascade_frontalface_default.xml"

		# Create the haar cascade
		faceCascade = cv2.CascadeClassifier(cascPath)

		# Read the image
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Detect faces in the image
		faces = faceCascade.detectMultiScale(
		    gray,
		    scaleFactor=1.1,
		    minNeighbors=5,
		    minSize=(30, 30)
		    #flags = cv2.CV_HAAR_SCALE_IMAGE
		)

		if len(faces) > 0:
			print '(face detected)'
		else:
			print '(face not detected)'
		return True if len(faces) > 0 else False

	# Returns tuple containing predicted value, actual value, and percent error.
	def getResult(self, image):
		# most recent image with full 'like' saturation
		imagePath = image['imagePath']
		# predict the number of likes
		yhat = int(self.predict(image))
		# actual number of likes
		y = image['likes']
		error = abs(float(yhat) - float(y)) / float(y)
		return (yhat, y, error)

# baseline = BaseLineAlgorithm()
# util.printBaselineResults(baseline)

data = util.getMetadata()
train, test = util.getDataSplit(data)

extractor = DummyFeatureExtractor()
model = LikenessV1()
model.train(train, extractor.extract)

cumulativeError = 0.0
for example in test:
	processedExample = extractor.extract(example)
	x = processedExample[0]
	y = processedExample[1]
	yhat = model.predict(x.reshape(1, -1))[0]
	print '---------------'
	print 'USER:', example['user']['username']
	print 'PREDICTED:', yhat
	print 'ACTUAL:', y
	print '---------------'
	cumulativeError += abs(y - yhat) / yhat
print 'AVERAGE ERROR (LikenessV1):', cumulativeError / len(test)