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
from sklearn import linear_model
from sklearn.decomposition import FastICA
from sklearn.feature_extraction import image  
from scipy import linalg

def main(mode):
    path = '/local/attale00/AFLW_ALL/'
    path_ea = '/local/attale00/AFLW_cropped/mouth_img_error/'
#    
    fileNames = utils.getAllFiles(path_ea);

    
    labs=utils.parseLabelFiles(path+'/labels/labels','mouth_opening',fileNames,cutoffSeq='.png',suffix='_face0.labels')
    
    testSet = fg.dataContainer(labs)
    components = 150
    roi=None
    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(40,120),roi=roi)
#    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(120,40),roi=roi,resizeFactor = .5)
# 
# perform ICA
    if mode not in ['s','v']:
        ica = FastICA(n_components=components,whiten=True)
        ica.fit(X)
        meanI=np.mean(X,axis=0)
        X1=X-meanI
        data=ica.transform(X1)
        filters=ica.components_
        
    elif mode in ['s','v']:
        W=np.load('/home/attale00/Desktop/classifiers/patches/filterMP1.npy')
        m=np.load('/home/attale00/Desktop/classifiers/patches/meanIMP1.npy')
        X1=X-m
        data=np.dot(X1,W.T)    
    
    for i in range(len(fileNames)):
            testSet.data[i].extend(data[i,:])
            
    print 'feature vector length: {}'.format(len(testSet.data[0]))

    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    rf=classifierUtils.standardRF(max_features = np.sqrt(len(testSet.data[0])),min_split=13,max_depth=40)
    #rf = svm.NuSVC()
    #rf = linear_model.SGDClassifier(loss='perceptron', eta0=1, learning_rate='constant', penalty=None)    
    if mode in ['s','v']:
        print 'Classifying with loaded classifier'
        _classifyWithOld(path,testSet,mode)
    elif mode in ['c']:
        print 'cross validation of data'
        rValues = classifierUtils.dissectedCV(rf,testSet)
        pickle.dump(rValues,open('errorpatch_ica','w'))
    elif mode in ['save']:
        print 'saving new classifier'
        _saveRF(testSet,rf,filters=filters,meanI=meanI)
    else:
        print 'not doing anything'
        
def _saveRF(testSet,rf,filters=None,meanI=None):
   
    rf.fit(testSet.data,testSet.targetNum)
    root='/home/attale00/Desktop/classifiers/thesis/errorpatches/'
    
    pickle.dump(rf,open(root+'errorpatch_ica','w'))
    
    f=open(root+'rficahogcolor.txt','w')
    f.write('Source Images: AFLWALL')
    f.write('attribute: Mouth')
    f.write('Features: ICA HOg color')
    f.write('100 comps \n')
    f.write('20 color bins \n')
    f.write('ppc 24,8, cpb 3,3 dir 5 \n')
    f.write('ROI:(50,74,96,160)\n')
 
    f.write('labels: none: 0, light,thick: 1\n')
    f.close()
    if filters is not None:
        np.save(root+'filter1',filters)
        np.save(root+'meanI1',meanI)
        


def _classifyWithOld(path,testSet,mode):
    #f=file('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_1','r')
    f=file('/home/attale00/Desktop/classifiers/thesis/errorpatches/errorpatch_hog','r')
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
        m='c'
            
    main(m)