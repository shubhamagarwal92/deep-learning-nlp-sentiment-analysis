import re
import numpy as np
import pandas as pd       
from bs4 import BeautifulSoup             

trainData = "labeledTrainData.tsv"
train = pd.read_csv(trainData, header=0, \
                    delimiter="\t", quoting=3)

reviewLength = 512
reviewsData = []
sentiments = []

raw_text = ''
## Pre-processing
for i in xrange(train.shape[0]):
	if i%1000 ==0:
		print i
	review = train['review'][i].strip()
	review = BeautifulSoup(str(review)).get_text()
	review = re.sub(' +',' ',review)
	review = review.lower()
	raw_text = raw_text + review + ' '
	reviewsData.append(review)
	sentiment = train['sentiment'][i]
	sentiments.append(sentiment)

reviewsData = np.asarray(reviewsData)
sentiments = np.asarray(sentiments)
dataDir = '/Users/admin/Documents/scripts/imdb/data/'
np.save(dataDir+'reviews.npy',reviewsData)
np.save(dataDir+'sentiments.npy',sentiments)
