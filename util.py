import json
import os
import fnmatch

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

# Gets all relevant metaData given a user's username.
def getMetadata(username):
	results = {}
	images = []
	with open('users/'+username+'/'+username+'.json') as json_data:
		data = json.load(json_data)
		for pic in data:
			if pic == data[0]:
				user = {}
				s = pic['user']['profile_picture']
				relativePath = './users/'+username
				user['profilePicPath'] = extractImagePathFromURL(s, relativePath)
				user['name'] = pic['user']['full_name']
				user['username'] = pic['user']['username']
				results['user'] = user
			imageData = {}
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
				images.append(imageData)

		results['images'] = images
	return results

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


# Gets average number of likes given a user's username
def getAverageLikes(username):
	sum = 0
	count = 0
	with open('users/'+username+'/'+username+'.json') as json_data:
		d = json.load(json_data)
		for pic in d:
			if pic and pic['likes'] and pic['likes']['count']:
				sum += pic['likes']['count']
				count += 1
	return float(sum) / float(count)

# Gets image paths of user
def getImagePaths(username):
	# From current directory
	relativePath = './users/'+username+'/'
	result = []
	for file in os.listdir(relativePath):
		if fnmatch.fnmatch(file, '*.jpg'):
			result.append(relativePath+file)
	return result

def printPredictorResults(predictor):
	cumulativeError = 0
	maxError = (0.0, '')
	minError = (1.0, '')
	usernames = sorted(getUsernames())
	for username in usernames:
		# sorted by most recent
		metadata = getMetadata(username)
		userData = metadata['user']
		imageData = sorted(metadata['images'], key=compareTimestamps)
		result = predictor.getResult(username, imageData)
		print(imageData[0]['name'], '(@'+username+')')
		print('Predicted:', result[0])
		print('Actual:', result[1])
		print('Error:', result[2])
		print('---------------')
		# sum errors
		cumulativeError += result[2]
		if result[2] > maxError[0]:
			maxError = (result[2], username)
		if result[2] < minError[0]:
			minError = (result[2], username)
	print('Average Error:', float(cumulativeError) / float(len(usernames)))
	print('Max Error:', maxError[0], '(@'+maxError[1]+')')
	print('Min Error:', minError[0], '(@'+minError[1]+')')
