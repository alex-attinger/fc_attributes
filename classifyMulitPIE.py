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

def main(mode):
    path='/local/attale00/'
    
    allFiles = utils.getAllFiles(path+'Multi-PIE/extracted')
    
    allLabelFiles = utils.getAllFiles(path+'Multi-PIE/labels')
    #allLabelFiles =  utils.getAllFiles(path+'a_labels')
    
    labeledImages = [i[0:16]+'.png' for i in allLabelFiles]
    
    labs=utils.parseLabelFiles(path+'/Multi-PIE/labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')
    #labs=utils.parseLabelFiles(path+'a_labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')
    
        
    
    testSet = fg.dataContainer(labs)
    
    
    roi = (0,32,0,64)
    #roi = (128,256,0,256)    
    
    eM=np.load('/home/attale00/Desktop/mouthMask.npy')
    m=cv2.resize(np.uint8(eM),(256,256));
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dil = cv2.dilate(m,strel)
    
    m=dil>0;

  
    fg.getHogFeature(testSet,roi,path=path+'Multi-PIE_grayScale64/',ending=None,extraMask = None)
    #fg.getColorHistogram(testSet,(50,190,110,402),path = path+'/Multi-PIE/extracted/',ending=None,colorspace='lab',range=(1.,255.0),bins = 20)
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    #testSet.targetNum = map(utils.mapGlassesLabels2Two,testSet.target)
    if mode in ['s','v']:
        print 'Classifying with loaded classifier'
        _classifyWithOld(path,testSet,mode)
    elif mode in ['c']:
        print 'cross validation of data'
        _cross_validate(testSet)
    elif mode in ['save']:
        print 'saving new classifier'
        _saveRF(testSet)
    else:
        print 'not doing anything'
        
def _saveRF(testSet):
    rf=classifierUtils.standardRF(max_features = np.sqrt(len(testSet.data[0])))
    rf.fit(testSet.data,testSet.targetNum)
    
    pickle.dump(rf,open('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_3','w'))
    
    f=open('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_4.txt','w')
    f.write('Source Images: Multi-Pie')
    f.write('attribute: Glasses')
    f.write('Features: Hog\n')
    f.write('Features: getHogFeature(mouthSet,roi2,path=path,ending=None,extraMask = None) on 256*256 grayScale with mouthmask dilated \n')
    f.write('ROI:(128,256,0,256)\n')
 
    f.write('labels: none: 0, light,thick: 1\n')
    f.close()
        
def _cross_validate(testSet):
    rf=classifierUtils.standardRF(max_features = np.sqrt(len(testSet.data[0])))
    print 'Scores'
    print classifierUtils.standardCrossvalidation(rf,testSet)
    print '----'
    return
    

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