# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:23:51 2013

@author: attale00
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import PyRoc

fn=['texture_mp_ica','texture_mp_hog','texture_mp_color']
labels=['ICA','Histogram of Oriented Gradients','Color Histogram']
c=['r','g','b','y','k']
av=[]
plt.figure()
plt.grid(True)
plt.title('Comparison')
stepsize=.05
x=np.arange(stepsize/2,1,stepsize)

for i,v in enumerate(fn):
    av=pickle.load(open(v))
    plt.plot(x,[np.average(a) for a in av],label = labels[i],color=c[i])

plt.xticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.yticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.xlim((0,1))
plt.ylim((0,1)) 
a=plt.gca()
a.set_aspect('equal')
plt.legend(loc=4)
plt.savefig('/home/attale00/Dropbox/tex_thesis/images/texture_ana/texture_cv_mp.pdf')

#######################################333



    
plt.show()