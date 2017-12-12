import numpy as np
import cv2
import util
from sklearn import linear_model
from featureExtractor import FeatureExtractor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
import os
import warnings
import json

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd") # lol
axis = np.s_[1:4]
n_clusters = 3

class LikenessV4():
	classifiers = []
	kmeansModel = None
	
	def train(self, data):
		X = []
		# print("training on " + str(len(data)) + " examples")
		cacheExists = os.path.isfile("cacheVectors.txt")
		xyPair = {}
		if cacheExists:
			with open("cacheVectors.txt", "r") as file:
			    for line in file:
			    	vector = [float(word) for word in line.split()]
			    	vector = np.asarray(vector)
			    	vector = np.delete(vector,axis,axis=0)
			    	for x in range(1,vector.shape[0]):
			    		vector[x] = vector[0]* vector[x]/50
					
			    	X.append(vector)
		else:
			print("cache doesn't exist. Can't run")
			quit()
			
		for i in range(len(data)):
			example = data[i]
			xyPair[tuple(X[i])] = example["likes"]
		
		self.kmeans = KMeans(n_clusters,n_init = 60,max_iter=700)

		self.kmeans = self.kmeans.fit(X)
		
		labels = self.kmeans.predict(X)
		X = np.asarray(X)

		clusters = [[] for i in range(n_clusters)]
		for i in range(len(labels)):
			vector = X[i]
			cIndex = labels[i]

			cluster = clusters[cIndex]
			cluster.append(vector)
			
			clusters[cIndex] = cluster
		# for i in range(n_clusters):
		# 	print len(clusters[i])
		
		for cluster in clusters:
			Y = []
			for vector in cluster:
				y = xyPair[tuple(vector)]
				Y.append(y)
			cluster = np.asarray(cluster)
			Y = np.asarray(Y)
			classifier = MLPRegressor(
				hidden_layer_sizes=(100, ), 
				activation='relu', 
				solver='adam', 
				alpha=0.0001, 
				batch_size=	'auto', 
				learning_rate='constant', 
				learning_rate_init=0.001, 
				power_t=0.5, 
				max_iter=5000, 
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
			classifier.fit(cluster, Y)
			self.classifiers.append(classifier)

	def predict(self, x):
		cIndex = self.kmeans.predict(x)[0]
		classifier = self.classifiers[cIndex]
		return classifier.predict(x)

data = util.getMetadata(readCache=True)
train, test = util.getDataSplit(data, readCache=True)
for i in range(1,10):
	n_clusters = i

	print "trying with " + str(n_clusters) + " clusters"
	
	model = LikenessV4()
	model.train(train)

	cumulativeError = 0.0
	cumError = 0.0
	allErrors = {}


	# print("testing on " + str(len(test)) + " examples")

	cacheExists = os.path.isfile("cacheTestVectors.txt")

	cache = []
	yCache = []

	if cacheExists:
		with open("cacheTestVectors.txt", "r") as file:
		    array = []
		    for line in file:
		    	cache.append([float(word) for word in line.split()])
		        array.append(line)

	cache = np.array(cache)
	cache = np.delete(cache,axis,axis=1)

	for i in range(len(test)):
		example = test[i]
		x = np.array(cache[i])
		y = example['likes']

		yhat = model.predict(x.reshape(1, -1))[0]
		cError = abs(y - yhat) / y
		cumError+= cError
		allErrors[example["imagePath"].encode('ascii','ignore')] = cError

	print 'AVERAGE ERROR:', cumError / len(test)
	print ''

	# with open('allErrors.txt', 'w') as file:
	#      file.write(json.dumps(allErrors))

