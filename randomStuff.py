# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:07:07 2013

@author: attale00

"""
#origin = '/local/attale00/Multi-PIE/labels/'
#dest = '/local/attale00/a_labels/'
#
#uniqueIDs=set()
#for i in allLabelFiles:
#    uniqueIDs.add(i[0:16])
#
#for j in allLabelFiles:
#    identifier = j[0:9]
#    #copy this file
#    shutil.copy(origin+j,dest)
#        
#        
#possibleMatches=[]
#for f in allFiles:
#    for l in allLabelFiles:
#        if f[0:9]==l[0:9]:
#            shutil.copyfile(origin+l,dest+f[0:-4]+'_face0.labels')
#    
#
#
#unlabeled = []
#
#for f in allFiles:
#    if f[0:16] not in uniqueIDs:
#        unlabeled.append(f)



from sklearn.utils import safe_asarray
import numpy as np
def balance_weights(y):
    """Compute sample weights such that the class distribution of y becomes
       balanced.

    Parameters
    ----------
    y : array-like
        Labels for the samples.

    Returns
    -------
    weights : array-like
        The sample weights.
    """
    y = safe_asarray(y)
    y = np.searchsorted(np.unique(y), y)
    bins = np.bincount(y)

    weights = 1. / bins.take(y)
    weights *= bins.min()

    return weights