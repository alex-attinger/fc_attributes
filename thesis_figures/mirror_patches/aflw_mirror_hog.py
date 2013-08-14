# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:25:06 2013

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
from scipy import linalg

def main(mode):
    path = '/local/attale00/AFLW_ALL/'
    path_ea = '/local/attale00/AFLW_cropped/cropped3/'
#    
    fileNames = utils.getAllFiles(path_ea);

    
    labs=utils.parseLabelFiles(path+'/labels/labels','mouth_opening',fileNames,cutoffSeq='.png',suffix='_face0.labels')
    
    
    roi=None
    testSet = fg.dataContainer(labs)
    testSetMirror = fg.dataContainer(labs)
    for f in range(len(testSetMirror.fileNames)):
        testSetMirror.fileNames[f]+='M'
    
  
    orientations = 9
    fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = orientations, cells_per_block=(3,3),pixels_per_cell=(24,8),maskFromAlpha=False)

  
    fg.getHogFeature(testSetMirror,roi,path='/local/attale00/AFLW_cropped/mirrored/', ending='.png',orientations = orientations, cells_per_block=(3,3),pixels_per_cell=(24,8))    
    
    testSet.addContainer(testSetMirror)
  
 
    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    rf=classifierUtils.standardRF(max_features = 30,min_split=12,max_depth=70)
    #rf = svm.NuSVC()
    #rf = linear_model.SGDClassifier(loss='perceptron', eta0=1, learning_rate='constant', penalty=None)    
    if mode in ['s','v']:
        print 'Classifying with loaded classifier'
        _classifyWithOld(path,testSet,mode)
    elif mode in ['c']:
        print 'cross validation of data'
        rValues = classifierUtils.dissectedCV(rf,testSet)
        pickle.dump(rValues,open('patches_cv_hog_{}'.format(orientations),'w'))
    elif mode in ['save']:
        print 'saving new classifier'
        _saveRF(testSet,rf)
    else:
        print 'not doing anything'
        
def _saveRF(testSet,rf,filters=None,meanI=None):
   
    rf.fit(testSet.data,testSet.targetNum)
    root='/home/attale00/Desktop/classifiers/thesis/mirror/'
    
    pickle.dump(rf,open(root+'rfHogMirror','w'))
    
    f=open(root+'rficahogcolormirror.txt','w')
    f.write('Source Images: AFLWALL Patches Mirror')
    f.write('attribute: Mouth')
    f.write('Features: ICA\n')
    f.write('100 comps \n')
    f.write('20 color bins \n')
    f.write('ppc 24,8, cpb 3,3 dir 5 \n')
    #f.write('ROI:(50,74,96,160)\n')
 
    f.write('labels: none: 0, light,thick: 1\n')
    f.close()
    if filters is not None:
        np.save(root+'filter1',filters)
        np.save(root+'meanI1',meanI)
        


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