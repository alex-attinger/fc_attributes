# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:19:39 2013

@author: attale00
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

bins=[10,20,40,60]
c=['r','g','b','y','k']

av=[]
plt.figure()
plt.grid(True)
plt.title('Color Histogram 3-fold Cross Validation')
stepsize=.05
x=np.arange(stepsize/2,1,stepsize)

for n,i in enumerate(bins):
    fn='patches_cv_color_{}'.format(i)
    av=pickle.load(open(fn))
    plt.plot(x,[np.average(a) for a in av],label='{} bins'.format(i),color=c[n])


plt.xticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.yticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.xlim((0,1))
plt.ylim((0,1)) 
a=plt.gca()
a.set_aspect('equal')
plt.legend(loc=4)
#plt.savefig('/home/attale00/Dropbox/tex_thesis/images/patches_cv_color.pdf')
#########################################################333333
orientations = [2,3,4,5,6,9]
c=['r','g','b','y','k','m','c']
av=[]
plt.figure()
plt.grid(True)
plt.title('Histogram of Oriented Gradients 3-fold Cross Validation')
stepsize=.05
x=np.arange(stepsize/2,1,stepsize)

for n,i in enumerate(orientations):
    fn='patches_cv_hog_{}'.format(i)
    av=pickle.load(open(fn))
    plt.plot(x,[np.average(a) for a in av],label='{} orientations'.format(i),color=c[n])


plt.xticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.yticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.xlim((0,1))
plt.ylim((0,1)) 
a=plt.gca()
a.set_aspect('equal')
plt.legend(loc=4)
#plt.savefig('/home/attale00/Dropbox/tex_thesis/images/patches_cv_hog.pdf')
#    
####################################################3
components = [5,10,25,35,50,100,150]
c=['r','g','b','y','k','m','c']
av=[]
plt.figure()
plt.grid(True)
plt.title('ICA 3-fold Cross Validation')
stepsize=.05
x=np.arange(stepsize/2,1,stepsize)

for n,i in enumerate(components):
    fn='patches_cv_ica_{}'.format(i)
    av=pickle.load(open(fn))
    plt.plot(x,[np.average(a) for a in av],label='{} components'.format(i),color=c[n])


plt.xticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.yticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.xlim((0,1))
plt.ylim((0,1)) 
a=plt.gca()
a.set_aspect('equal')
plt.legend(loc=4)
#plt.savefig('/home/attale00/Dropbox/tex_thesis/images/patches_cv_ica.pdf')

########################
fn=['patches_cv_ica_35','patches_cv_hog_9','patches_cv_color_40','patches_cv_combined']
labels=['ICA','Histogram of Oriented Gradients','Color Histogram','Combination']

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
plt.savefig('/home/attale00/Dropbox/tex_thesis/images/patches_ana/patches_cv_comparison.pdf')
    
plt.show()