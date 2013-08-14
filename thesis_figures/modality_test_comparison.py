
import pickle
import matplotlib.pyplot as plt
import numpy as np
import PyRoc


fn=['patches/mp_combined','texture/mp_combined','errorpatches/errorpatch_test_mode_0','modality_texture_patch']
labels=['Image Patch','Extracted Texture','Error Patch','Combination']
c=['r-','g-','b-','y-','k-']
av=[]
stepsize=.05
x=np.arange(stepsize/2,1,stepsize)
roclist = []
for i,v in enumerate(fn):
    r=pickle.load(open(v))
    r.linestyle=c[i]
    roclist.append(r)
PyRoc.plot_multiple_roc(roclist,title='Comparing image modality',labels=labels,plot_average=False)
plt.xticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.yticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.legend(loc=4)
#plt.savefig('/home/attale00/Dropbox/tex_thesis/images/modality_combo.pdf')
#######################################################################################################
fn=['modality_combo','modality_errorpatch_patch','modality_texture_patch','modality_errorpatch_texture']
labels=['Errorpatch + Patch + Texture','Errorpatches + Patches','Texture + Patches','Errorpatches + Texture']
c=['y-','g-','b-','r-','k-']
av=[]
stepsize=.05
x=np.arange(stepsize/2,1,stepsize)
roclist = []
for i,v in enumerate(fn):
    r=pickle.load(open(v))
    r.linestyle=c[i]
    roclist.append(r)
PyRoc.plot_multiple_roc(roclist,title='Comparing image modality',labels=labels,plot_average=False)
plt.xticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.yticks(np.arange(0.1,1.1,0.1),np.arange(0.1,1.1,0.1))
plt.xlim((0.0,0.3))
plt.ylim((0.7,1.0))
plt.legend(loc=4)

plt.show()