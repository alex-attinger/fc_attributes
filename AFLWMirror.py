# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:25:06 2013

@author: attale00
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:29:01 2013

@author: attale00
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:24:39 2013

This script classifies the multipie pictures with the random forest classifer learned on the aflw database pics
. It was trained on hog features only on the down scaled images. see the accompanying info file to the classfier for details

@author: attale00
"""

import utils
import featureGeneration as fg
import cv2
import numpy as np
import pickle
import sys
import plottingUtils
import classifierUtils
from sklearn import svm
from sklearn.decomposition import FastICA  
from scipy import linalg

def main(mode):
    path = '/local/attale00/AFLW_ALL'
    path_ea = path+'/color128/'
    
    fileNames = utils.getAllFiles(path_ea);
    
    
    
    
    labs=utils.parseLabelFiles(path+'/labels/labels','mouth_opening',fileNames,cutoffSeq='.png',suffix='_face0.labels')
    
    
    
    testSet = fg.dataContainer(labs)
    
    testSetMirror = fg.dataContainer(labs)
    for f in range(len(testSetMirror.fileNames)):
        testSetMirror.fileNames[f]+='M'
    
    
    roi=(50,74,96,160)
 
 

    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(128,256),roi=roi)
    Y=fg.getAllImagesFlat(path+'/mirror128/',testSet.fileNames,(128,256),roi=roi)
    Z=np.concatenate((X,Y),axis=0)
    # perform ICA
    ica = FastICA(n_components=100,whiten=True)
    ica.fit(Z)
    meanI=np.mean(Z,axis=0)
    X1=X-meanI
    Y1=Y-meanI    
    data=ica.transform(X1)
    datam=ica.transform(Y1)
    filters=ica.components_
    for i in range(len(fileNames)):
        testSet.data[i].extend(data[i,:])
        testSetMirror.data[i].extend(datam[i,:])


    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = 3, cells_per_block=(6,2),maskFromAlpha=False)
    #fg.getColorHistogram(testSet,roi,path=path_ea,ending='.png',colorspace='lab',bins=10)

  
    #pca
#    n_samples, n_features = X.shape
#
#    mean_ = np.mean(X, axis=0)
#    X -= mean_
#    U, S, V = linalg.svd(X)
#    explained_variance_ = (S ** 2) / n_samples
#    explained_variance_ratio_ = (explained_variance_ /explained_variance_.sum())
#    K=V / S[:, np.newaxis] * np.sqrt(n_samples)
#    filters=K[:100]
#    data=np.dot(X,filters.T)    
    
    testSet.addContainer(testSetMirror)
            
    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    rf=classifierUtils.standardRF(max_features = np.sqrt(len(testSet.data[0])),min_split=5,max_depth=40)
    if mode in ['s','v']:
        print 'Classifying with loaded classifier'
        _classifyWithOld(path,testSet,mode)
    elif mode in ['c']:
        print 'cross validation of data'
        classifierUtils.dissectedCV(rf,testSet)
    elif mode in ['save']:
        print 'saving new classifier'
        _saveRF(testSet,rf,filters=filters,meanI=meanI)
    else:
        print 'not doing anything'
        
def _saveRF(testSet,rf,filters=None,meanI=None):
   
    rf.fit(testSet.data,testSet.targetNum)
    root='/home/attale00/Desktop/classifiers/ica/'
    
    pickle.dump(rf,open(root+'rf128ICAM','w'))
    
    f=open(root+'rf128ica.txt','w')
    f.write('Source Images: AFLWALL with mirrors')
    f.write('attribute: Mouth')
    f.write('Features: ICA')
    f.write('100 comps \n')
    f.write('ROI:(50,74,96,160)\n')
 
    f.write('labels: none: 0, light,thick: 1\n')
    f.close()
    if filters is not None:
        np.save(root+'filter2',filters)
        np.save(root+'meanI2',meanI)
        


def _classifyWithOld(path,testSet,mode):
    #f=file('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_1','r')
    f=file('/home/attale00/Desktop/classifiers/SVMMouth_1','r')
    clf = pickle.load(f)
    testSet.classifiedAs=clf.predict(testSet.data)
    testSet.hasBeenClassified = True
    if mode =='s':
        _score(clf,testSet)
    else:
        _view(clf,testSet,path+'Multi-PIE/extracted/')
        _score(clf,testSet)
     

def _score(clf,testSet):
    score = clf.score(testSet.data,testSet.targetNum)
    testSet.hasBeenClassified = True
    classifierUtils.evaluateClassification(testSet,{0:'closed or narrow',1:'open or wide open'})
    print 'Overall Score: {:.3f}'.format(score)
    return
def _view(clf,testSet,path):
    viewer = plottingUtils.ClassifiedImViewer(path,testSet)
    viewer.view(comparer=plottingUtils.MouthTwo2FourComparer)
    
    
if __name__=='__main__':
    if len(sys.argv)==2:
        if sys.argv[1] in ['s','Score']:
            m = 's'
        elif sys.argv[1] in ['v','View']:
            m='v'
        elif sys.argv[1] in ['c']:
            m='c'
        elif sys.argv[1] in ['save']:
            m='save'
        else:
            print 'Option not supported, valid options are s,v. Now just scoring'
            m='s'
    else:
        m='s'
            
    main(m)