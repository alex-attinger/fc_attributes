# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:19:39 2013

@author: attale00
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import PyRoc

mode=[0,1,2]
l=['Mean and Std','Mean only','Std only']
c=['r','g','b','y','k']

av=[]
plt.figure()
plt.grid(True)
plt.title('Error patches 3-fold Cross Validation')
stepsize=.05
x=np.arange(stepsize/2,1,stepsize)

for n,i in enumerate(mode):
    fn='errorpatch_mode_{}'.format(i)
    av=pickle.load(open(fn))
    plt.plot(x,[np.average(a) for a in av],label=l[n],color=c[n])
av=pickle.load(open('errorpatch_hog'))
plt.plot(x,[np.average(a) for a in av],label='HOG',color=c[3])
av=pickle.load(open('errorpatch_ica'))
plt.plot(x,[np.average(a) for a in av],label='ICA',color=c[4])
plt.xticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.yticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.xlim((0,1))
plt.ylim((0,1)) 
a=plt.gca()
a.set_aspect('equal')
plt.legend(loc=4)
#plt.savefig('/home/attale00/Dropbox/tex_thesis/images/errorpatch_ana/errorpatches_cv_stats.pdf')
#########################################################333333
plt.figure()
r=pickle.load(open('errorpatch_test_mode_0'))
r.plot(title='Errorpatches',include_baseline=True)
#plt.savefig('/home/attale00/Dropbox/tex_thesis/images/errorpatch_train.pdf')
##########################################3
overlap=[0,1,2]
l=['0','1','2']
c=['r','g','b','y','k']

av=[]
plt.figure()
plt.grid(True)
plt.title('Error patches 3-fold Cross Validation Overlap')
stepsize=.05
x=np.arange(stepsize/2,1,stepsize)

for n,i in enumerate(overlap):
    fn='errorpatch_overlap_{}'.format(i)
    av=pickle.load(open(fn))
    plt.plot(x,[np.average(a) for a in av],label=l[n],color=c[n])


plt.xticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.yticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.legend(loc=4)
plt.xlim((0,1))
plt.ylim((0,1)) 
a=plt.gca()
a.set_aspect('equal')
#plt.savefig('/home/attale00/Dropbox/tex_thesis/images/errorpatch_ana/errorpatches_cv_overlap.pdf')
#####################################################333
sz=[2,4,8,20,40]
l=['(6,2)','(12,4)','(24,8)','(60,20)','(120,40)']
c=['r','g','b','y','k']

av=[]
plt.figure()
plt.grid(True)
plt.title('Error patches 3-fold Cross Validation Size')
stepsize=.05
x=np.arange(stepsize/2,1,stepsize)

for n,i in enumerate(sz):
    print i
    fn='errorpatch_size_{}'.format(i)
    av=pickle.load(open(fn))
    plt.plot(x,[np.average(a) for a in av],label=l[n],color=c[n])


plt.xticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.yticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.legend(loc=4)
plt.xlim((0,1))
plt.ylim((0,1)) 
a=plt.gca()
a.set_aspect('equal')

plt.savefig('/home/attale00/Dropbox/tex_thesis/images/errorpatch_ana/errorpatches_cv_size.pdf')
plt.show()