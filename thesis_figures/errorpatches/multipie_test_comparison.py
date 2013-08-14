import pickle
import matplotlib.pyplot as plt
import numpy as np
import PyRoc


fn=['errorpatch_test_mode_0','errorpatch_test_mode_1','errorpatch_test_mode_2','errorpatch_test_hog','errorpatch_test_ica']
labels=['Mean and STD','Mean Only','STD only','HOG','ICA']
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
plt.savefig('/home/attale00/Dropbox/tex_thesis/images/errorpatch_ana/errorpatch_train.pdf')
plt.show()