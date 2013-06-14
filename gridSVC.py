print __doc__

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: Simplified BSD

from pprint import pprint
from time import time
import logging
import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import featureGeneration as fg
import utils
from sklearn.decomposition import FastICA
import numpy as np 
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


###############################################################################
#data
path = '/local/attale00/AFLW_ALL/'
path_ea = '/local/attale00/AFLW_cropped/cropped2/'
#    
fileNames = utils.getAllFiles(path_ea);


labs=utils.parseLabelFiles(path+'/labels/labels','mouth_opening',fileNames,cutoffSeq='.png',suffix='_face0.labels')



testSet = fg.dataContainer(labs)


roi=(0,37,0,115)

 
X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(37,115),roi=roi)
 
#        
# perform ICA
ica = FastICA(n_components=100,whiten=True)
ica.fit(X)
meanI=np.mean(X,axis=0)
X1=X-meanI
data=ica.transform(X1)
filters=ica.components_
for i in range(len(fileNames)):
    testSet.data[i].extend(data[i,:])

testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)



###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
clf = SVC()

parameters = {'C': list(np.arange(1.0, 20.0,2.0)),
                  'class_weight': ['auto'],
                'kernel':['rbf','poly'],
                'gamma':[0.0,0.5,0.9],
                'degree':range(1,7,2)}

#if __name__ == "__main__":
# multiprocessing requires the fork to happen in a __main__ protected
# block

# find the best parameters for both the feature extraction and the
# classifier
grid_search = GridSearchCV(clf, parameters, n_jobs=-1, verbose=1)

print "Performing grid search..."

print "parameters:"
pprint(parameters)
t0 = time()
grid_search.fit(data, np.array(testSet.targetNum))
print "done in %0.3fs" % (time() - t0)
print

print "Best score: %0.3f" % grid_search.best_score_
try:
    print grid_search.best_estimator_
except E:
    print 'printing did not work'
pickle.dump(grid_search.best_score_,open('/local/attale00/clf','w'))
