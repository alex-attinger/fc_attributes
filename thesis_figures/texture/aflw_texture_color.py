# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:25:06 2013

@author: attale00
"""

# -*- coding: utf-8 -*-


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
    
    
    roi=(50,74,96,160)
    #roi=(44,84,88,168)    
    
    
#    eM=np.load('/home/attale00/Desktop/mouthMask.npy')
#    m=cv2.resize(np.uint8(eM),(256,256));
#    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#    dil = cv2.dilate(m,strel)
#    
#    m=dil>0;


    path_mp = '/local/attale00/extracted_pascal__4__Multi-PIE/color128/'
    mpFiles = utils.getAllFiles(path_mp)
            
 
    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(128,256),roi=roi)
    #Y=fg.getAllImagesFlat(path_mp,mpFiles,(128,256),roi=roi)
    #Z=np.concatenate((X,Y),axis=0)
    Z=X
#        
    # perform ICA
#    ica = FastICA(n_components=50,whiten=True)
#    ica.fit(Z)
#    meanI=np.mean(X,axis=0)
#    
#    X1=X-meanI
#    data=ica.transform(X1)
#    filters=ica.components_
#    for i in range(len(fileNames)):
#        testSet.data[i].extend(data[i,:])

    bins = 40

    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = 3, cells_per_block=(6,2),maskFromAlpha=False)
    fg.getColorHistogram(testSet,roi,path=path_ea,ending='.png',colorspace='lab',bins=bins)

  

            
    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    rf=classifierUtils.standardRF(max_features = np.sqrt(len(testSet.data[0])),min_split=1,max_depth=40)
    if mode in ['s','v']:
        print 'Classifying with loaded classifier'
        _classifyWithOld(path,testSet,mode)
    elif mode in ['c']:
        print 'cross validation of data'
        rValues = classifierUtils.dissectedCV(rf,testSet)
        pickle.dump(rValues,open('texture_color_{}'.format(bins),'w'))
    elif mode in ['save']:
        print 'saving new classifier'
        _saveRF(testSet,rf,filters=None,meanI=None)
    else:
        print 'not doing anything'
        
        
    return
        
def _saveRF(testSet,rf,filters=None,meanI=None):
   
    rf.fit(testSet.data,testSet.targetNum)
    root='/home/attale00/Desktop/classifiers/thesis/'
    
    pickle.dump(rf,open(root+'color_40','w'))
    
    f=open(root+'rf128ica.txt','w')
    f.write('Source Images: AFLWALL')   
    f.write('attribute: Mouth')
    f.write('Features: ICA')
    f.write('100 comps \n')
    f.write('ROI:(50,74,96,160)\n')
 
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