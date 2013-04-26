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
import classifierUtils
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
    #labs=utils.parseLabelFiles(path+'/mouth_labels/labels','glasses',fileNames,cutoffSeq='.png',suffix='_face0.labels')

    print('-----computing Features-----')
    #make 10 bin hist for each mouth
    #roi = (40,200,100,200)
    roi = (50,190,110,402) 
    roi2=(0,128,0,256)
    roi=(0,64,0,128)
    #roi2=(128,256,0,256)
    mouthSet = fg.dataContainer(labs)
    #fg.getHistogram(20,roi,hrange=(0,255),dataC = mouthSet,path = path+'/extracted/gradients/Direction/',ending='_0.png')
    eM=np.load('/home/attale00/Desktop/mouthMask.npy')
    m=cv2.resize(np.uint8(eM),(256,256));
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dil = cv2.dilate(m,strel)
    
    
    m=dil>0;
#    em=m[roi[0]:roi[1],roi[2]:roi[3]]
#    m= m !=True
  
    fg.getHogFeature(mouthSet,roi2,path=path_ea+'/grayScale128/',ending='_0.png',extraMask = None)
    #fg.getPixelValues(mouthSet,roi,path=path_ea+'/',ending='_0.png',mask =m, scaleFactor = 10)    
    #fg.getColorHistogram(mouthSet,roi,path=path_ea+'/',ending='_0.png',colorspace=None,range=(1.0,255.0),bins = 20)   
    mouthSet.targetNum=map(utils.mapMouthLabels2Two,mouthSet.target)
    #mouthSet.targetNum=map(utils.mapGlassesLabels2Two,mouthSet.target)
    
    
    score=[]
    frac=np.arange(0.2,1.0,.05)
    for i in frac:
        trainingSet,testSet=mouthSet.splitInTestAndTraining(frac=i)
        rf=classifierUtils.standardRF(max_features = np.sqrt(len(mouthSet.data[0])))
        rf.fit(trainingSet.data,trainingSet.targetNum)
        score.append(rf.score(testSet.data,testSet.targetNum))
        testSet.hasBeenClassified=True
        testSet.classifiedAs=rf.predict(testSet.data)
        print '---------------- {} -----------'.format(i)
        classifierUtils.evaluateClassification(testSet,{0:'closed',1:'open'})
    plt.plot(frac,score,'-*')
    plt.show()        
  
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
