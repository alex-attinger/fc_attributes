# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:33:09 2013

@author: attale00
"""

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def doGridSearch(data,classLabel,parameters):
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(clf, parameters, n_jobs=-1, verbose=1)
    grid_search.fit(data, classLabel)
    print "Best score: %0.3f" % grid_search.best_score_
    try:
        print grid_search.best_estimator_
    except Exception:
        print 'printing did not work'
        
    return (grid_search.best_score_,grid_search.best_estimator_)