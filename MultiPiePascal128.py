

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

def main(mode):
    path = '/local/attale00/extracted_pascal__4__Multi-PIE'
    path_ea = path+'/color128/'
   
    allLabelFiles =  utils.getAllFiles('/local/attale00/a_labels')
    
    labeledImages = [i[0:16]+'.png' for i in allLabelFiles]
    
    #labs=utils.parseLabelFiles(path+'/Multi-PIE/labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')
    labs=utils.parseLabelFiles('/local/attale00/a_labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')
    
    
    #fileNames = utils.getAllFiles(path_ea);
    
    
    
    
    #labs=utils.parseLabelFiles(path+'/labels/labels','mouth_opening',fileNames,cutoffSeq='.png',suffix='_face0.labels')
    
    
    
    testSet = fg.dataContainer(labs)
    
    
    roi=(50,74,96,160)
    #roi=(44,84,88,168)    
    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(128,256),roi=roi)
    W=np.load('/home/attale00/Desktop/classifiers/ica/filter1.npy')
    m=np.load('/home/attale00/Desktop/classifiers/ica/meanI1.npy')
    X-=m
    data=np.dot(X,W.T)
    for i in range(len(testSet.fileNames)):
        testSet.data[i].extend(data[i,:])
#    eM=np.load('/home/attale00/Desktop/mouthMask.npy')
#    m=cv2.resize(np.uint8(eM),(256,256));
#    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#    dil = cv2.dilate(m,strel)
#    
#    m=dil>0;

    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

  
    fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = 3, cells_per_block=(6,2),maskFromAlpha=False)
    fg.getColorHistogram(testSet,roi,path=path_ea,ending='.png',colorspace='lab',bins=10)    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    
    rf=classifierUtils.standardRF(max_features = np.sqrt(len(testSet.data[0])),min_split=5,max_depth=40)    
    if mode in ['s','v']:
        print 'Classifying with loaded classifier'
        _classifyWithOld(path,testSet,mode)
    elif mode in ['c']:
        print 'cross validation of data'
        print 'Scores'
        #print classifierUtils.standardCrossvalidation(rf,testSet,n_jobs=5)
        #_cvDissect(testSet,rf)
        classifierUtils.dissectedCV(rf,testSet)        
        print '----'
       
    elif mode in ['save']:
        print 'saving new classifier'
        _saveRF(testSet)
    else:
        print 'not doing anything'
        
def _saveRF(testSet,rf):
    
    rf.fit(testSet.data,testSet.targetNum)
    
    pickle.dump(rf,open('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_nTSmallHOGColor','w'))
    
    f=open('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_nTHOGColor.txt','w')
    f.write('Source Images: Multi-Pie')
    f.write('attribute: Glasses')
    f.write('Features: Hog\n')
    f.write('Features: getHogFeature(orientations = 4, cells_per_block=(26,9),maskFromAlpha=True \n')
    f.write('ROI:(88,165,150,362)\n')
 
    f.write('labels: none: 0, light,thick: 1\n')
    f.close()
        


def _classifyWithOld(path,testSet,mode):
    #f=file('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_1','r')
    #np.random.shuffle(testSet.data)    
    cp='/home/attale00/Desktop/classifiers/ica/rf128ICAHOGCOLOR'
    print 'using classifier: '+cp
    f=file(cp,'r')
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

def _cvDissect(testSet,rf,mapping):
    for i in range(3):
        train,test = testSet.splitInTestAndTraining(frac=.7)
        rf.fit(train.data,train.targetNum)
        test.classifiedAs=rf.predict(test.data)
        test.hasBeenClassified = True
        classifierUtils.evaluateClassification(test,{0:'closed or narrow',1:'open or wide open'})
    
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