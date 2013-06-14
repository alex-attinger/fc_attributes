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
    
    labs=utils.parseLabelFiles(path+'/labels/labels','mouth_opening',fileNames,cutoffSeq='.png',suffix='_face0.labels')
    
    
    
    testSet = fg.dataContainer(labs)
    testSetMirror = fg.dataContainer(labs)
    for f in range(len(testSetMirror.fileNames)):
        testSetMirror.fileNames[f]+='M'
    
    roi=None    
    #roi=(44,84,88,168)    
    
    
#    eM=np.load('/home/attale00/Desktop/mouthMask.npy')
#    m=cv2.resize(np.uint8(eM),(256,256));
#    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#    dil = cv2.dilate(m,strel)
#    
#    m=dil>0;


            
 
    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(40,120),roi=roi)
    Y=fg.getAllImagesFlat('/local/attale00/AFLW_cropped/mirrored/',testSet.fileNames,(40,120),roi=roi)
    Z=np.concatenate((X,Y),axis=0)
    
    # perform ICA
    ica = FastICA(n_components=100,whiten=True)
    ica.fit(Z)
    meanI=np.mean(Z,axis=0)
    X1=X-meanI
    Y1=Y-meanI    
    data=ica.transform(X1)
    datam=ica.transform(Y1)
    filters=ica.components_
    for i in range(len(fileNames)):
        testSet.data[i].extend(data[i,:])
        testSetMirror.data[i].extend(datam[i,:])
#        



    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = 5, cells_per_block=(3,3),pixels_per_cell=(24,8),maskFromAlpha=False)
    
    fg.getColorHistogram(testSet,roi,path=path_ea,ending='.png',colorspace='lab',bins=20)
    #mirror part
    fg.getHogFeature(testSetMirror,roi,path='/local/attale00/AFLW_cropped/mirrored/', ending='.png',orientations = 5, cells_per_block=(3,3),pixels_per_cell=(24,8))    
    fg.getColorHistogram(testSetMirror,roi,path='/local/attale00/AFLW_cropped/mirrored/',ending = '.png',colorspace='lab',bins=20)
        
    testSet.addContainer(testSetMirror)
  
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
    rf=classifierUtils.standardRF(max_features = 17,min_split=7,max_depth=40,n_estimators=90)
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
    
    pickle.dump(rf,open(root+'rfICAHogColorMirror','w'))
    
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