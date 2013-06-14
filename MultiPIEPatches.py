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
from scipy import linalg

def main(mode):
    path = '/local/attale00/AFLW_ALL/'
    path_ea = '/local/attale00/AFLW_cropped/multiPIE_cropped3/'
    

   
    allLabelFiles =  utils.getAllFiles('/local/attale00/a_labels')
    
    labeledImages = [i[0:16]+'.png' for i in allLabelFiles]
    
    #labs=utils.parseLabelFiles(path+'/Multi-PIE/labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')
    labs=utils.parseLabelFiles('/local/attale00/a_labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')
        
    
#    
    fileNames = labeledImages;
#    minr = 10000;
#    for f in fileNames:
#        im = cv2.imread(path_ea+f,-1)
#        if im.shape[0]!=40 or im.shape[1]!=120:
#            print f
#            print im.shape
#        minr = minr if im.shape[0]>= minr else im.shape[0]
#    
#    print minr
#    
    
   
    
    
    testSet = fg.dataContainer(labs)
    
    
    roi=(0,37,0,115)
    roi=None    
    #roi=(44,84,88,168)    
    
    
#    eM=np.load('/home/attale00/Desktop/mouthMask.npy')
#    m=cv2.resize(np.uint8(eM),(256,256));
#    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#    dil = cv2.dilate(m,strel)
#    
#    m=dil>0;


            
 
    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(40,120),roi=roi)
 
        
    # perform ICA
    if mode not in ['s','v']:
        ica = FastICA(n_components=100,whiten=True)
        ica.fit(X)
        meanI=np.mean(X,axis=0)
        X1=X-meanI
        data=ica.transform(X1)
        filters=ica.components_
        
    elif mode in ['s','v']:
        W=np.load('/home/attale00/Desktop/classifiers/patches/filter1.npy')
        m=np.load('/home/attale00/Desktop/classifiers/patches/meanI1.npy')
        X1=X-m
        data=np.dot(X1,W.T)    
    
    for i in range(len(fileNames)):
            testSet.data[i].extend(data[i,:])

    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = 5, pixels_per_cell=(24,8),cells_per_block=(3,3),maskFromAlpha=False)
    fg.getColorHistogram(testSet,roi,path=path_ea,ending='.png',colorspace='lab',bins=20)

  
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
    
   
            
    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    rf=classifierUtils.standardRF(max_features = 27,min_split=13,max_depth=40)
    #rf = svm.NuSVC()
    #rf = linear_model.SGDClassifier(loss='perceptron', eta0=1, learning_rate='constant', penalty=None)    
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
    root='/home/attale00/Desktop/classifiers/patches/'
    
    pickle.dump(rf,open(root+'rfICAMultiPie','w'))
    
    f=open(root+'rficahogcolormirrorMP.txt','w')
    f.write('Source Images: AFLWALL Patches Mirror')
    f.write('attribute: Mouth')
    f.write('Features: ICA\n')
    f.write('100 comps \n')
    #f.write('20 color bins \n')
    #f.write('ppc 24,8, cpb 3,3 dir 5 \n')
    #f.write('ROI:(50,74,96,160)\n')
 
    f.write('labels: none: 0, light,thick: 1\n')
    f.close()
    if filters is not None:
        np.save(root+'filterMP1',filters)
        np.save(root+'meanIMP1',meanI)
        


def _classifyWithOld(path,testSet,mode):
    #f=file('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_1','r')
    f=file('/home/attale00/Desktop/classifiers/patches/rfICAHogColorMirror','r')
    print 'classifier used: '+ f.name
    clf = pickle.load(f)
    testSet.classifiedAs=clf.predict(testSet.data)
    testSet.hasBeenClassified = True
    if mode =='s':
        _score(clf,testSet)
    else:
        _view(clf,testSet,'/local/attale00/AFLW_cropped/multiPIE_cropped3/')
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