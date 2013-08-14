
import pickle
import matplotlib.pyplot as plt
import numpy as np
import PyRoc


#fn=[,'mp_hog9','mp_color40','mp_combined','combined2']
#labels=['ICA 35','ICA 150','Histogram of Oriented Gradients','Color Histogram','Combination','Combo 2']

fn=['mp_hog9','mirrored','color_only']
labels=['Original Dataset','Mirrored Dataset','color only']

c=['r-','g-','b-','y-','k-','c-']
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
#plt.savefig('/home/attale00/Dropbox/tex_thesis/images/mirror/hog_test.pdf')
plt.show()