import numpy as np
import cv2
import util
from sklearn import linear_model
from featureExtractor import FeatureExtractor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import os
import warnings
import json

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd") # lol
axis = np.s_[1:4]
# axis = 2
# axis  =1:4
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
	# Trains the model. Takes all data and extracts 15-vectors from the posts.
	# Runs Minibatch Gradient Descent to optimize and return a weight vector.
	def train(self, data, featureExtractor,cacheData):
		X = []
		Y = []
		print("training on " + str(len(data)) + " examples")
		count = 0
		cacheExists = os.path.isfile("cacheVectors.txt")

		if cacheExists:
			with open("cacheVectors.txt", "r") as file:
			    for line in file:
			    	X.append([float(word) for word in line.split()])
			# with open("yCache.txt", "r") as file:
			#     for line in file:
			#     	Y.append(float(line))
			
		for i in range(len(data)):
			example = data[i]
			if not cacheExists:
				output = featureExtractor(example)

				X.append(output[0])
			Y.append(example["likes"])
			if count%100 == 0:
				print(str(count*1.0/len(data)) + "\% done!")
			count+=1

		X = np.asarray(X)
		Y = np.asarray(Y)
		
		# print(X)

		if not cacheExists:
			cacheData()
			with open("yCache.txt", 'w') as f:
			    for y in Y:
			        f.write(str(_string))
			        f.write('\n')

		
		print("Done with Feature Extraction with X and Y vectors of size: ")
		print(X.shape)
		print(Y.shape)

		X= np.array(X)
		X = np.delete(X,axis,axis=1)
		
		self.classifier = linear_model.LinearRegression(fit_intercept=True, copy_X=False)
		self.classifier.fit(X, Y)
		print "parameters", self.classifier.coef_

	def predict(self, x):
		return self.classifier.predict(x)

class LikenessV2():
	theta = np.random.random(15)
	classifier = None

	# Trains the model. Takes all data and extracts 15-vectors from the posts.
	# Runs Minibatch Gradient Descent to optimize and return a weight vector.
	def train(self, data, featureExtractor,cacheData):
		X = []
		Y = []
		print("training on " + str(len(data)) + " examples")
		count = 0
		for example in data[0:50]:
			# print(example)
			output = featureExtractor(example)
			# print(output)
			X.append(output[0])
			Y.append(output[1])

			if count%50 == 0:
				print(str(count*1.0/len(data)) + "\% done!")
			count+=1

		X = np.asarray(X)
		Y = np.asarray(Y)

		cacheData()

		print(X.shape)
		print(Y.shape)

		self.classifier = SVR(kernel='poly', C=1e3, degree=2)
		self.classifier.fit(X, Y)
		self.theta = self.classifier.coef_
		return self.theta

	def predict(self, x):
		return self.classifier.predict(x)
		

class LikenessV3():
	classifier = None

	# Trains the model. Takes all data and extracts 15-vectors from the posts.
	# Runs Minibatch Gradient Descent to optimize and return a weight vector.
	def train(self, data, featureExtractor,cacheData):
		X = []
		Y = []
		print("training on " + str(len(data)) + " examples")
		count = 0
		cacheExists = os.path.isfile("cacheVectors.txt")

		if cacheExists:
			with open("cacheVectors.txt", "r") as file:
			    for line in file:
			    	X.append([float(word) for word in line.split()])
			# with open("yCache.txt", "r") as file:
			#     for line in file:
			#     	Y.append(float(line))
			
		for i in range(len(data)):
			example = data[i]
			if not cacheExists:
				output = featureExtractor(example)

				X.append(output[0])
			Y.append(example["likes"])
			if count%100 == 0:
				print(str(count*1.0/len(data)) + "\% done!")
			count+=1

		X = np.asarray(X)
		Y = np.asarray(Y)
		if not cacheExists:
			cacheData()
			with open("yCache.txt", 'w') as f:
			    for y in Y:
			        f.write(str(_string))
			        f.write('\n')

		
		print("Done with Feature Extraction with X and Y vectors of size: ")
		print(X.shape)
		print(Y.shape)

		self.classifier = MLPRegressor(
			hidden_layer_sizes=(100, ), 
			activation='relu', 
			solver='adam', 
			alpha=0.0001, 
			batch_size=275, 
			learning_rate='constant', 
			learning_rate_init=0.001, 
			power_t=0.5, 
			max_iter=500, 
			shuffle=True, 
			random_state=1, 
			tol=0.0001, 
			verbose=False, 
			warm_start=False, 
			momentum=0.9, 
			nesterovs_momentum=True, 
			early_stopping=False, 
			validation_fraction=0.1, 
			beta_1=0.9, 
			beta_2=0.999, 
			epsilon=1e-08
		)
		
		X= np.array(X)
		# X[1:, 1:,] = X[:, 0] * X[:, 1:,]   
		for y in range(X.shape[0]):
			for x in range(1,X.shape[1]):
				X[y][x] = X[y][0]* X[y][x]/50

		X = np.delete(X,axis,axis=1)
		self.classifier.fit(X, Y)

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

data = util.getMetadata(readCache=True)
train, test = util.getDataSplit(data, readCache=True)

extractor = FeatureExtractor()
model = LikenessV3()
model.train(train, extractor.extract,extractor.cacheData)

cumulativeError = 0.0
cumError = 0.0
allErrors = {}


print("testing on " + str(len(test)) + " examples")

cacheExists = os.path.isfile("cacheTestVectors.txt")
cache = []
yCache = []
if cacheExists:
	with open("cacheTestVectors.txt", "r") as file:
	    array = []
	    for line in file:
	    	cache.append([float(word) for word in line.split()])
	        array.append(line)
	# with open("yTestCache.txt", "r") as file:
	#     for line in file:
	#     	Y.append(float(line))
cache = np.array(cache)
		
cache = np.delete(cache,axis,axis=1)
# print(cache.shape)
for i in range(len(test)):
	example = test[i]
	x = None
	if cacheExists:
		x = np.array(cache[i])
	else:
		processedExample = extractor.extract(example)
		x = processedExample[0]
	y = example['likes']

	yhat = model.predict(x.reshape(1, -1))[0]

	# print '---------------'
	# print 'USER:', example['user']['username']
	# print 'PREDICTED:', yhat
	# print 'ACTUAL:', y
	# print '---------------'
	error = abs(y - yhat) / yhat
	cError = abs(y - yhat) / y
	cumulativeError += error
	cumError+= cError
	allErrors[example["imagePath"].encode('ascii','ignore')] = error

if not cacheExists:
	extractor.cacheData(cachingTest=True)	

print 'AVERAGE ERROR:', cumulativeError / len(test)
print 'AVERAGE ERROR:', cumError / len(test)

with open('allErrors.txt', 'w') as file:
     file.write(json.dumps(allErrors))

