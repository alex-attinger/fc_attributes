# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:25:41 2013

@author: attale00
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:35:54 2013

@author: attale00
"""

## mouth

import utils
import numpy as np
import featureGeneration as fg
import classifierUtils

from sklearn.ensemble import RandomForestClassifier

import cv2
import pickle

def main(nJobs = 1):

    path = '/local/attale00/GoodPose/extracted_alpha/grayScale64'
    
    fileNames = utils.getAllFiles(path);
    

    labs=utils.parseLabelFiles('/local/attale00/GoodPose'+'/mouth_labels','mouth',fileNames,cutoffSeq='_0.png',suffix='_face0.labels')
    print('-----computing Features-----')

    roi2 = (0,32,0,64)
    mouthSet = fg.dataContainer(labs)

    #load the mask for the mouth room pixels and dilate it
    eM=np.load('/home/attale00/Desktop/mouthMask.npy')
    m=cv2.resize(np.uint8(eM),(256,256));
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dil = cv2.dilate(m,strel)

    m=dil>0;

    #get the features
    fg.getHogFeature(mouthSet,roi2,path=path+'/',ending=None,extraMask = None)
    
    #map the string labels to numbers (required by sklearn)
    #change the mapping here for different classifiers
    mouthSet.targetNum=map(utils.mapMouthLabels2Two,mouthSet.target)
    n_estimators = 100
    min_split = 10
    max_depth = 20
    max_features = np.sqrt(len(mouthSet.data[0]))
    
    rf = classifierUtils.standardRF(max_features = max_features)
    rf2=classifierUtils.standardRF(max_features=max_features)
    
    score=classifierUtils.standardCrossvalidation(rf2,mouthSet)

    rf.fit(mouthSet.data,mouthSet.targetNum)
    
    pickle.dump(rf,open('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_12','w'))
    
    f=open('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_12.txt','w')
    f.write('Trained on aflw\n')
    f.write('Attribute: mouth' )
    f.write('Features: getHogFeature(mouthSet,roi2,path=path,ending=None,extraMask = m) on 64*64 grayScale 3 direction bins \n')
    f.write('ROI:(0,32,0,64)\n')
    f.write('labels: closed, narrow: 0, open, wideOpen: 1\n')
    f.write('CV Score: {}\n'.format(score))
    f.close()
    
    

        
if __name__=='__main__':

    main()
