# likeness
An application that predicts post popularity from the features of a given Instagram post.

Notes:

11/15
- Now the dataset is cached (so it doesn't have to read in a bunch of different JSON files every time).
- There is now a consistent train/test data split.
- The algorithm now runs scikit linear regression model.

11/14
- Just installed `scikit-learn` (which depends upon `numpy` and `scipy`) and I don't understand virtualenv.
- Pipenv just stopped working. Attempted to uninstall everything and reinstall everything. Nothing works.
- Starting to look at the project again for the upcoming progress report.
- Current framework: Linear Regression using squared loss
- Color and other basic photo qualities are important.
	- Source: https://www.curalate.com/blog/6-image-qualities-that-drive-more-instagram-likes/
- Faces are important.
	- Source: http://www.news.gatech.edu/2014/03/20/face-it-instagram-pictures-faces-are-more-popular
- The captions are important.
	- Specifically, the number of emojis and hashtags.
	- Source: https://sproutsocial.com/insights/instagram-stats/
	- Length is not important, but @mentions are.
	- Source: http://get.simplymeasured.com/rs/simplymeasured2/images/InstagramStudy2014Q3.pdf?mkt_tok=3RkMMJWWfF9wsRolua%252FAZKXonjHpfsX57%252BwtX6a2lMI%252F0ER3fOvrPUfGjI4CTsViI%252BSLDwEYGJlv6SgFQrDEMal41bgNWRM%253D
- Timing is important.
	- Specifically, time of day and day of week.
	- Source: https://blog.bufferapp.com/instagram-stats-instagram-tips

10/26
- Downloaded OpenCV!
- Working on the infrastructure to analyze images in bulk.
- Possible cause of skew: new pictures (< 1 week old). New pictures may not be fully saturated. Their like counts could still be in flux.
- Baseline algorithm is implemented as follows:
	- Input: a user's profile and a photo.
	- A mean is calculated across the like counts of the 50 most recent photos.
	- A coefficient is calculated based on the features of the photo. For the baseline, this coefficient is either 1.1 (if face is detected) or 0.9 (if face is not detected).
	- Output: coefficient multiplied by the mean.
- Baseline algorithm has 46.60% average error rate. This means that on average, the algorithm overestimated or underestimated the like count for a given image of a user by 46.60%. The human error rate was ~30%, so there is a large amount of avoidable bias to overcome.
- Possible ideas:
	- Belly button detection
	- Proportion of photo that is face
	- If user's face is in photo

10/25
- Scraped the most recent ~50 photos of the top 40 instagram celebrities using `instagram-scraper`. Ignored @taylorswift because she only had 25 photos. Sad.
	- Source: https://github.com/rarcega/instagram-scraper
- We say we have approximately 50 photos because we are excluding videos.