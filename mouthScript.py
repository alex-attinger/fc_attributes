# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:35:54 2013

@author: attale00
"""

## mouth

import utils
import numpy as np
import featureGeneration as fg
from sklearn import svm, cross_validation
from sklearn.ensemble import RandomForestClassifier
import sys
import plottingUtils
import matplotlib.pyplot as plt
import cv2

def main(nJobs = 1):

    path = '/local/attale00/GoodPose'
    path_ea = path+'/extracted_alpha'
    path_adata = path_ea + '/a_data'
    
    fileNames = utils.getAllFiles(path+'/targets');
    
    attribute = 'mouth'
    
    attribute_values = utils.parseLabelINIFile(path+'/mouth_labels/labels.ini',attribute);
    
    print('------------Attribute: \t'+attribute+' ---------------')
    for i in attribute_values:
        print('Value: \t'+i)
        
    print('----------------------------')
    print('----------parsing label files------')
    labs=utils.parseLabelFiles(path+'/mouth_labels','mouth',fileNames,cutoffSeq='.png',suffix='_face0.labels')
    print('-----computing Features-----')
    #make 10 bin hist for each mouth
    #roi = (40,200,100,200)
    roi = (50,190,110,402) 
    roi2 = (0,128,0,256)
    mouthSet = fg.dataContainer(labs)
    #fg.getHistogram(20,roi,hrange=(0,255),dataC = mouthSet,path = path+'/extracted/gradients/Direction/',ending='_0.png')
    eM=np.load('/home/attale00/Desktop/mouthMask.npy')
    m=cv2.resize(np.uint8(eM),(256,256));
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dil = cv2.dilate(m,strel)
    
    
    m=dil>0;
#    em=m[roi[0]:roi[1],roi[2]:roi[3]]
#    m= m !=True
  
    fg.getHogFeature(mouthSet,roi2,path=path_ea+'/grayScaleSmall/',ending='_0.png',extraMask = m)
    #fg.getPixelValues(mouthSet,roi,path=path_ea+'/',ending='_0.png',mask =m, scaleFactor = 10)    
        
    mouthSet.targetNum=map(utils.mapMouthLabels2Two,mouthSet.target)
    n_estimators = range(10,160,20);
    max_features = range(2,22,2)
    max_depth = range(5,40,5)
    max_depth.append(100)
    
    min_split = range(1,20,2)
    
    score=[]
    var = []
    for n in n_estimators:    
        scoresRF = _crossValidate(mouthSet, max_depth = 20,n_estimators =n ,nJobs = nJobs,max_features = np.sqrt(len(mouthSet.data[0])),min_split = 5)
   
        score.append(scoresRF.mean())
        var.append(scoresRF.std())
        
    print scoresRF
    plt.errorbar(n_estimators,score,yerr=var)
    plt.xlabel('number of trees')
    plt.ylabel('cross val score')
    
    
    plt.show()        
    
    #classifier
    #linSVM = svm.SVC(kernel = 'linear',C=1)
    
    #this takes forever: check if that can be true
    #scoresLinSVM = cross_validation.cross_val_score(linSVM,data,y=targetNum,n_jobs=-1,verbose = 1)
    
    #implement random forest classifier with verbosity level
#    roi_narrow=(60,160,130,382)
#    extraMask = np.load('/home/attale00/Desktop/emptyMouthMask.npy')
#    
#    fg.getMeanAndVariance(roi_narrow,mouthSet,path_ea+'/',extraMask = extraMask,ending='_0.png')
#    scoresRF = _crossValidate(mouthSet,max_features = 13)
#    print 'Orientation and mean and cov' +str(scoresRF)    
    return
    
def _crossValidate(dataSet,nJobs = 1,n_estimators = 60, max_features=7,max_depth = None,min_split = 1):
    print '-----------cross validation-------------'    
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features =max_features ,max_depth=max_depth,min_split=min_split, random_state=0,n_jobs=1)    
    #cv = cross_validation.ShuffleSplit(len(dataSet.targetNum), n_iter=3,test_size=0.3, random_state=0)
    scoresRF = cross_validation.cross_val_score(rf,dataSet.data,y=dataSet.targetNum,n_jobs=nJobs,cv=3)
    return scoresRF

def _classify(trainingSet, testSet,plotting = False,n_estimators=60,max_features=3):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features =max_features ,max_depth=None,min_split=1, random_state=0,n_jobs=1)    
    
    rf=rf.fit(trainingSet.data,trainingSet.targetNum)
   
    print(rf.score(testSet.data,testSet.targetNum))
    out = rf.predict(testSet.data);
    targetNum = np.array(testSet.targetNum);
    totalNum =float(len(targetNum))
    tp = np.count_nonzero(out==targetNum)
    
    tpr=tp/totalNum
    print(str(tp) + ' ' +str(totalNum))
    print(tpr)
    
    fp = np.count_nonzero((out==1) & (targetNum == 0))
    fpr=fp/totalNum
    fn = totalNum-fp-tp  
    fnr = fn/totalNum
    print('Fraction Correct: {0:.2f}\nfraction fals open: {1:.2f} \n fraction false closed: {2:.2f} '.format(tpr,fpr,fnr))
    if (plotting):    
        testSet.classifiedAs = map(utils.two2MouthLabel,out)
        viewer=plottingUtils.ClassifiedImViewer(path_ea+'/grayScale/',testSet,suffix='_0.png')
        viewer.view(comparer=plottingUtils.MouthTwo2FourComparer,grayScale=True)
    return (tpr,fpr,fnr)


        
if __name__=='__main__':
    njobs = sys.argv[1]
    main(int(njobs))
