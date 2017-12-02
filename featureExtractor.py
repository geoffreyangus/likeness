import numpy as np
import cv2
import re
import csv
import datetime
from skimage import color
from PorterStemmer import PorterStemmer

class FeatureExtractor():

	def extract(self,jsonBlob):
		imagePath = jsonBlob['imagePath']
		self.img = cv2.imread(imagePath)
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		self.caption = jsonBlob['caption']
		self.timeStamp = jsonBlob['timestamp']
		self.features = []

		self.stemmer = PorterStemmer()
		reader = csv.reader(open('sentiment.txt', 'rb'))
		rawSentiment = dict(reader)

		# Convert the words to their stemd equivalents, which will make comparison
		# of input strings easier later
		self.sentiment = {}
		for word in rawSentiment:
			self.sentiment[self.stemmer.stem(word)] = rawSentiment[word]
		return (np.array(self.getFeatures()),jsonBlob['likes'])

    # Will run sentiment detection on the entire input passed in, so if we want
    # to ignore movie names, make sure to pass in a string with these stripped
	def getSentiment(self, input):
		# separate punction out
		input = input.replace(","," ,")
		# stem all the words to make sure we're dealing with a consistent set of tokens
		words = input.lower().split()
		words = [self.stemmer.stem(w) for w in words]

		# Rate this input positive vs. negative
		posScore = 0
		negScore = 0
		#info for handling negation words eg. 'not', 'pierce', 'didn't' and handling contradicting words eg. 'but', 'geoff','however'
		negation = False
		contradiction = False
		contradictions = {'but', 'however', 'although', 'though', 'yet', 'except', 'nevertheless', 'nonetheless', 'despite','tho'}
		negationWords = {'not', 'never', 'neither', 'nor', 'isn\'t', 'didn\'t', 'wasn\'t', 'hasn\'t','isnt', 'didnt', 'wasnt', 'hasnt'}
		punctuation = {'.', '?', '!'}

		for w in words:
			if w in contradictions: # reset logic after contradictions, since they infer that what came before doesn't matter to the sentiment
				posScore,negScore = 0,0
				contradiction = True
			if w in negationWords:
				negation = True
			if w in punctuation:
				negation = False
			if w == ",":
				if contradiction:
					posScore,negScore = 0,0
				contradiction = False
			if w not in self.sentiment: continue
			if negation: # things after a negation tend to have a higher overall weight
				if self.sentiment[w] == "pos": negScore += 2
				else: posScore += 2
			else:
				if self.sentiment[w] == "pos": posScore += 1
				else: negScore += 1
		possyVibesOnly = [0,0] 
		if posScore > negScore: 
			possyVibesOnly = [1,0]
		else: 
			possyVibesOnly = [0,1]
		return possyVibesOnly

	def dominantColors(self):
		img = self.img
		arr = np.float32(img)
		pixels = arr.reshape((-1, 3))

		n_colors = 4
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .15)
		flags = cv2.KMEANS_RANDOM_CENTERS
		_, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 2, flags)

		# print(centroids)
		# print(centroids)
		centroids = np.array([centroids]).astype(int)
		hsvs = color.rgb2hsv(centroids/255.0)
		colors = [0]*8
		for hsv in hsvs[0]:
			degree = hsv[0]*360
			# print(degree)
			# print(abs(degree//45))
			colors[int(degree//45)]+=1
		return colors

	def getFeatures(self):
		# self.features += self.gradients()
		self.features += [self.numMentions()]
		self.features += [self.numHashtags()]
		self.features += [self.commentLength()]
		self.features += self.timeStampInfo()
		self.features += [self.numFaces()]
		self.features += [self.brightness()]
		self.features += self.dominantColors()
		self.features += self.getSentiment(self.caption)

		# print(len(self.features))
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
		day = [0]*7
		hour = [0]*24
		day[date.weekday()] = 1
		hour[date.hour] = 1
		return day + hour

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