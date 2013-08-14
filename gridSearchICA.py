# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:38:39 2013

@author: attale00
"""

from pprint import pprint
import logging
import pickle
import featureGeneration as fg
import utils
from sklearn.decomposition import FastICA
import numpy as np 
import matplotlib.pyplot as plt
import GridSearchUtils
# Display progress logs on stdout



def main():

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    
    
    ###############################################################################
    #data
    path = '/local/attale00/AFLW_ALL/'
    path_ea = '/local/attale00/AFLW_cropped/cropped3/'
    
    fileNames = utils.getAllFiles(path_ea);
    
    
    labs=utils.parseLabelFiles(path+'/labels/labels','mouth_opening',fileNames,cutoffSeq='.png',suffix='_face0.labels')
    
    
    
    testSet = fg.dataContainer(labs)
    

     
    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(40,120),roi=None)
     
    #        
    # perform ICA
    ica = FastICA(n_components=250,whiten=True)
    ica.fit(X)
    meanI=np.mean(X,axis=0)
    X1=X-meanI
    data=ica.transform(X1)

    for i in range(len(fileNames)):
        testSet.data[i].extend(data[i,:])

    
    DATA=np.zeros((len(testSet.data),len(testSet.data[0])))
    
    for i in range(len(testSet.data)):
        DATA[i,:]=np.array(testSet.data[i])
    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
        
    classLabels = np.array(testSet.targetNum)
    
    scores = []
    
    
    parameters = {'n_estimators': range(90, 260,20),
                  'max_depth': range(50, 110,20),
                'min_samples_split':range(5,35,5),
                'max_features':range(25,75,10),
                'min_samples_leaf':range(1,10,5)}

#    parameters = {'n_estimators': range(90, 100,20),
#                  'max_depth': range(50, 60,20),
#                'min_samples_split':range(5,15,5),
#                'max_features':range(25,35,10),
#                'min_samples_leaf':range(1,10,5)}
                
    print "Performing grid search..."

    print "parameters:"
    pprint(parameters)
    
    for i in range(80,260,15):
        mf = i if i<75 else 75
        parameters['max_features']=range(10,mf,10)
        s=GridSearchUtils.doGridSearch(data[:,0:i],classLabels,parameters)
        temp = (s[0],s[1],i)
        scores.append(temp)
    
    scores_s=sorted(scores,key = lambda x:x[0])
    
    print scores_s
    
    pickle.dump(scores_s,open('/local/attale00/gridsearches/ICA_3','w'))
    
    
    
    
    
    
if __name__=='__main__':
    main()
