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

from sklearn.ensemble import RandomForestClassifier

import cv2
import pickle

def main(nJobs = 1):

    path = '/local/attale00/GoodPose/extracted_alpha/grayScaleSmall'
    
    fileNames = utils.getAllFiles(path);
    

    labs=utils.parseLabelFiles('/local/attale00/GoodPose'+'/mouth_labels','mouth',fileNames,cutoffSeq='_0.png',suffix='_face0.labels')
    print('-----computing Features-----')

    roi2 = (0,128,0,256)
    mouthSet = fg.dataContainer(labs)

    #load the mask for the mouth room pixels and dilate it
    eM=np.load('/home/attale00/Desktop/mouthMask.npy')
    m=cv2.resize(np.uint8(eM),(256,256));
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dil = cv2.dilate(m,strel)

    m=dil>0;

    #get the features
    fg.getHogFeature(mouthSet,roi2,path=path+'/',ending=None,extraMask = m)
    
    #map the string labels to numbers (required by sklearn)
    #change the mapping here for different classifiers
    mouthSet.targetNum=map(utils.mapMouthLabels2Two,mouthSet.target)
    n_estimators = 100
    min_split = 10
    max_depth = 20
    max_features = np.sqrt(len(mouthSet.data[0]))
    
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features =max_features ,max_depth=max_depth,min_split=min_split, random_state=0,n_jobs=1)    

    rf.fit(mouthSet.data,mouthSet.targetNum)
    
    pickle.dump(rf,open('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_1','w'))
    
    f=open('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_1.txt','w')
    f.write('Features: getHogFeature(mouthSet,roi2,path=path,ending=None,extraMask = m) on 256*256 grayScale with mouthmask dilated \n')
    f.write('ROI:(0,128,0,256)\n')
    f.write('scaled and dilated mouthMask.npy\n')
    f.write('labels: closed, narrow: 0, open, wideOpen: 1\n')
    f.close()
    
    

        
if __name__=='__main__':

    main()
