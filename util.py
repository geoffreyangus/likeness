import json

usernameList = open('ig_users.txt').read().splitlines()

# Gets all captions given a user's username.
def getCaptions(username):
	results = []

	with open('users/'+username+'/'+username+'.json') as json_data:
	    d = json.load(json_data)
	    for pic in d:
	    	if pic and pic['caption'] and pic['caption']['text']:
	    		results.append(pic['caption']['text'])

	return results

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


for username in usernameList:
	print username, getAverageLikes(username)