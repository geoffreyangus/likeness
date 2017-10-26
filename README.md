# likeness
An application that predicts post popularity from the features of a given Instagram post.

Notes:

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

10/25
- Scraped the most recent ~50 photos of the top 40 instagram celebrities using `instagram-scraper`. Ignored @taylorswift because she only had 25 photos. Sad.
	- Source: https://github.com/rarcega/instagram-scraper
- We say we have approximately 50 photos because we are excluding videos.