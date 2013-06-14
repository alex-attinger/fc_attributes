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
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import featureGeneration as fg
import utils
from sklearn.decomposition import FastICA
import numpy as np 
import matplotlib.pyplot as plt
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


###############################################################################
#data
path = '/local/attale00/AFLW_ALL/'
path_ea = '/local/attale00/AFLW_cropped/cropped3/'
#path_ea = '/local/attale00/AFLW_ALL/color128/'
#    
fileNames = utils.getAllFiles(path_ea);


labs=utils.parseLabelFiles(path+'/labels/labels','mouth_opening',fileNames,cutoffSeq='.png',suffix='_face0.labels')



testSet = fg.dataContainer(labs)


roi=(0,37,0,115)
roi=None
#roi=(50,74,96,160)
 
X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(40,120),roi=roi)
 
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

fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = 5, cells_per_block=(3,3),pixels_per_cell=(24,8),maskFromAlpha=False)
#
fg.getColorHistogram(testSet,roi,path=path_ea,ending='.png',colorspace='lab',bins=20)

DATA=np.zeros((len(testSet.data),len(testSet.data[0])))

for i in range(len(testSet.data)):
    DATA[i,:]=np.array(testSet.data[i])

testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)




###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
clf = RandomForestClassifier()

parameters = {'n_estimators': range(10, 120,10),
                  'max_depth': range(5, 50,5),
                'min_samples_split':range(5,80,2),
                'max_features':range(3,51,3),
                'min_samples_leaf':range(1,31,2)}

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
grid_search.fit(DATA, np.array(testSet.targetNum))
print "done in %0.3fs" % (time() - t0)
print

print "Best score: %0.3f" % grid_search.best_score_
try:
    print grid_search.best_estimator_
except Exception:
    print 'printing did not work'
#d=[]
#for i in range(len(grid_search.grid_scores_)):
#    d.append(grid_search.grid_scores_[i][1])
#
#plt.plot(range(len(d)),d,'*')
#plt.show()
pickle.dump(grid_search.param_grid,open('/local/attale00/paramgrid1','w'))
pickle.dump(grid_search.grid_scores_,open('local/attale00/gridscores1','w'))
pickle.dump(grid_search.best_score_,open('/local/attale00/clfICAHOGCOLOR','w'))
