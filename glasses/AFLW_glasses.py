# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:25:06 2013

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
import matplotlib.pyplot as plt

def main(mode):
    labelFiles='/local/attale00/aflw_original_labels'
    path_ea = '/local/attale00/AFLW_cropped/eyes/'
    

    fileNames = utils.getAllFiles(path_ea)
    goods = []
    for f in fileNames:
        im = cv2.imread(path_ea+f)
        if im.shape[0] == 40 and im.shape[1] ==120:
            goods.append(f)
    
    fileNames = goods
    labs=utils.parseLabelFiles(labelFiles,'glasses',fileNames,cutoffSeq='.png',suffix='.labels')
    
    
   
    
    
    testSet = fg.dataContainer(labs)
    
    


            
# 
#    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(40,120),roi=roi)
# 
#        
#    # perform ICA
#    if mode not in ['s','v']:
#        ica = FastICA(n_components=100,whiten=True)
#        ica.fit(X)
#        meanI=np.mean(X,axis=0)
#        X1=X-meanI
#        data=ica.transform(X1)
#        filters=ica.components_
#        
#    elif mode in ['s','v']:
#        W=np.load('/home/attale00/Desktop/classifiers/patches/filter2.npy')
#        m=np.load('/home/attale00/Desktop/classifiers/patches/meanI2.npy')
#        X1=X-m
#        data=np.dot(X1,W.T)    
#    
#    for i in range(len(testSet.fileNames)):
#            testSet.data[i].extend(data[i,:])
#
#    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fg.getHogFeature(testSet,None,path=path_ea,ending='.png',extraMask = None,orientations = 5, pixels_per_cell=(24,8),cells_per_block=(3,3),maskFromAlpha=False)

    #testSet.targetNum=map(lambda x: 1 if x=='light' else 0,testSet.target)
   
    #rf = svm.NuSVC()
    #rf = linear_model.SGDClassifier(loss='perceptron', eta0=1, learning_rate='constant', penalty=None)    
    if mode in ['s','v']:
        print 'Classifying with loaded classifier'
        obj=classifierUtils.classifyWithOld(path,testSet,mode,clfPath = '/home/attale00/Desktop/classifiers/thesis/color_only')       
        #pickle.dump(obj,open('color_only','w'))
    elif mode in ['c']:
        print 'cross validation of data'
        
        
        testSet.targetNum=map(lambda x: 1 if x=='1' else 0,testSet.target)
        rf=classifierUtils.standardRF(max_features = 27,min_split=5,max_depth=60,n_estimators = 500)
        rValues = classifierUtils.dissectedCV(rf,testSet)
        plt.title('No glasses against rest')
        plt.show()
        
        #pickle.dump(rValues,open('patches_mp_','w'))
    elif mode in ['save']:
        print 'saving new classifier'
        testSet.targetNum=map(lambda x: 1 if x=='1' else 0,testSet.target)
        rf=classifierUtils.standardRF(max_features = 27,min_split=5,max_depth=60,n_estimators = 500)
        _saveRF(testSet,rf)
    else:
        print 'not doing anything'
        
def _saveRF(testSet,rf,filters=None,meanI=None):
   
    rf.fit(testSet.data,testSet.targetNum)
    root='/home/attale00/Desktop/classifiers/thesis/glasses/'
    
    pickle.dump(rf,open(root+'rfHog','w'))
    
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
        


#def _classifyWithOld(path,testSet,mode):
#    #f=file('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_1','r')
#    f=file('/home/attale00/Desktop/classifiers/patches/rfICAHogColor','r')
#    print 'classifier used: '+ f.name
#    clf = pickle.load(f)
#    testSet.classifiedAs=clf.predict(testSet.data)
#    testSet.probabilities=clf.predict_proba(testSet.data)
# 
#    for i in range(len(testSet.data)):
#        if testSet.probabilities[i][1]>=.35:
#            testSet.classifiedAs[i]=1
#        else:
#            testSet.classifiedAs[i]=0
#    testSet.hasBeenClassified = True
#    print classifierUtils.splitByPose(testSet)
#    if mode =='s':
#        _score(clf,testSet)
#    else:
#        _view(clf,testSet,'/local/attale00/AFLW_cropped/multiPIE_cropped3/')
#        _score(clf,testSet)
#     
#
#def _score(clf,testSet):
#    score = clf.score(testSet.data,testSet.targetNum)
#    testSet.hasBeenClassified = True
#    classifierUtils.evaluateClassification(testSet,{0:'closed or narrow',1:'open or wide open'})
#    print 'Overall Score: {:.3f}'.format(score)
#    return
#def _view(clf,testSet,path):
#    viewer = plottingUtils.ClassifiedImViewer(path,testSet)
#    viewer.view(comparer=plottingUtils.MouthTwo2FourComparer)
#    

    
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