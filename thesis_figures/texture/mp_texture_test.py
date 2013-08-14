
import pickle
import matplotlib.pyplot as plt
import numpy as np
import PyRoc


fn=['mp_ica50','mp_hog4','mp_color40','mp_combined']
labels=['ICA','Histogram of Oriented Gradients','Color Histogram','Combination']
c=['r-','g-','b-','y-','k-']
av=[]
stepsize=.05
x=np.arange(stepsize/2,1,stepsize)
roclist = []
for i,v in enumerate(fn):
    r=pickle.load(open(v))
    r.linestyle=c[i]
    roclist.append(r)
PyRoc.plot_multiple_roc(roclist,title='Comparison',labels=labels,plot_average=False)
plt.xticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.yticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.legend(loc=4)
#plt.savefig('/home/attale00/Dropbox/tex_thesis/images/texture_test_mp.pdf')
########################3333333333

fn=['mp_hog4','hog_pose']
labels=['HOG only','HOG and Pose']
c=['r-','g-','b-','y-','k-']
av=[]
stepsize=.05
x=np.arange(stepsize/2,1,stepsize)
roclist = []
for i,v in enumerate(fn):
    r=pickle.load(open(v))
    r.linestyle=c[i]
    roclist.append(r)
PyRoc.plot_multiple_roc(roclist,title='Comparison',labels=labels,plot_average=False)
plt.xticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.yticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.legend(loc=4)
#plt.savefig('/home/attale00/Dropbox/tex_thesis/images/texture_test_mp.pdf')

plt.show()