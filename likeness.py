import cv2
import util

class BaseLineAlgorithm():
	# Predicts the like count for a given user
	def predict(self, username, imagePath):
		avg = util.getAverageLikes(username)
		coefficient = self.processImage(imagePath)
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
			print('(face detected)')
		else:
			print('(face not detected)')
		return True if len(faces) > 0 else False

	# Returns tuple containing predicted value, actual value, and percent error.
	def getResult(self, username, metadata):
		# most recent image with full 'like' saturation
		imagePath = metadata[-2]['imagePath']
		# predict the number of likes
		yhat = int(self.predict(username, imagePath))

		# actual number of likes
		y = metadata[-2]['likes']
		error = abs(float(yhat) - float(y)) / float(y)
		return (yhat, y, error)

baseline = BaseLineAlgorithm()
util.printPredictorResults(baseline)
