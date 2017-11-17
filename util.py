import json
import os
import fnmatch
import numpy as np

PROCESSED_DATA_PATH = './data_processed.json'
TRAIN_PATH = './train.json'
TEST_PATH = './test.json'
from featureExtractor import FeatureExtractor

# Gets the usernames from the ig_users.txt file
def getUsernames():
	result = []
	with open('ig_users.txt') as users:
		result = users.read().splitlines()
	return result

# Compares timestamps of images. To be used with list.sort or sorted.
def compareTimestamps(item):
	return item['timestamp']

# extracts image filename from raw URL
def extractImagePathFromURL(url, newRelativePath):
	index = url.rfind('/')
	return newRelativePath + url[index:]

# Gets all relevant metadata given a user's username.
def getMetadata(readCache = True):
	# If file already cached, simply extract data from the file
	if readCache and os.path.isfile(PROCESSED_DATA_PATH) and os.path.getsize(PROCESSED_DATA_PATH) > 0:
		return getMetadataFromFile()

	result = []
	usernames = sorted(getUsernames())
	for username in usernames:
		result += getMetadataByUsername(username)

	# Save newly processed data
	with open('data_processed.json', 'w+') as f:
		json.dump(result, f, indent=4)
	return result

def getMetadataFromFile():
	with open(PROCESSED_DATA_PATH) as json_data:
		return json.load(json_data)

# Gets all relevant metadata given a user's username.
def getMetadataByUsername(username):
	images = []
	with open('users/'+username+'/'+username+'.json') as json_data:
		data = json.load(json_data)
		
		likeSum = 0
		commentSum = 0
		validPicCount = 0
		user = {}
		for pic in data:
			imageData = {}
			if pic == data[0]:
				s = pic['user']['profile_picture']
				relativePath = './users/'+username
				user['profilePicPath'] = extractImagePathFromURL(s, relativePath)
				user['fullName'] = pic['user']['full_name']
				user['username'] = pic['user']['username']
			imageData['user'] = user
			if validatePic(pic):
				s = pic['images']['low_resolution']['url']
				relativePath = './users/'+username
				imageData['imagePath'] = extractImagePathFromURL(s, relativePath)
				imageData['name'] = pic['user']['full_name']
				imageData['timestamp'] = pic['created_time']
				if not (pic['caption'] and pic['caption']['text']):
					imageData['caption'] = ''
				else:
					imageData['caption'] = pic['caption']['text']
				imageData['likes'] = pic['likes']['count']

				likeSum += imageData['likes']
				commentSum += len(imageData['caption'])
				validPicCount += 1

				images.append(imageData)

		likeAvg = float(likeSum) / float(validPicCount)
		commentAvg = float(commentSum) / float(validPicCount)

		for image in images:
			image['user']['averageLikes'] = likeAvg
			image['user']['averageCommentLength'] = commentAvg

	return images

# Verifies that the json object is well-formed
def validatePic(pic):
	if not pic:
		return False
	if not (pic['type'] and pic['type'] == 'image'):
		return False
	if not (pic['user'] and pic['user']['full_name']):
		return False
	if not (pic['created_time']):
		return False
	if not (pic['likes'] and pic['likes']['count']):
		return False
	if not (pic['images'] and pic['images']['low_resolution']):
		return False
	return True

# Gets image paths of user
def getImagePaths(username):
	# From current directory
	relativePath = './users/'+username+'/'
	result = []
	for file in os.listdir(relativePath):
		if fnmatch.fnmatch(file, '*.jpg'):
			result.append(relativePath+file)
	return result

def getDataSplit(data, readCache=True):
	if readCache and os.path.isfile(PROCESSED_DATA_PATH) and os.path.getsize(PROCESSED_DATA_PATH) > 0:
		train = []
		test = []
		with open(TRAIN_PATH) as json_data:
			train = json.load(json_data)
		with open(TEST_PATH) as json_data:
			test = json.load(json_data)
		return (train, test)

	return util.generateNewDataSplit(data)

def generateNewDataSplit(data):
	newTestDataset = list(np.random.choice(data, int(round(len(data) * 0.2)), replace=False))
	with open(TEST_PATH, 'w+') as f:
		json.dump(newTestDataset, f, indent=4)
	newTrainDataset = [x for x in data if x not in newTestDataset]
	with open(TRAIN_PATH, 'w+') as f:
		json.dump(newTrainDataset, f, indent=4)
	return (newTrainDataset, newTestDataset)

def printBaselineResults(predictor):
	cumulativeError = 0
	maxError = (0.0, '')
	minError = (1.0, '')
	metadata = getMetadata()
	test = getDataSplit(metadata)[1]
	for example in test:
		# sorted by most recent
		result = predictor.getResult(example)
		print example['user']['username']
		print 'Predicted:', result[0]
		print 'Actual:', result[1]
		print 'Error:', result[2]
		print '---------------'\
		# sum errors
		cumulativeError += result[2]
		if result[2] > maxError[0]:
			maxError = (result[2], example['user']['username'])
		if result[2] < minError[0]:
			minError = (result[2], example['user']['username'])
	print 'Average Error:', float(cumulativeError) / float(len(test))
	print 'Max Error:', maxError[0], '(@'+maxError[1]+')'
	print 'Min Error:', minError[0], '(@'+minError[1]+')'

