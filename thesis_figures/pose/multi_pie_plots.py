import pickle
import matplotlib.pyplot as plt
import numpy as np
import PyRoc


bins = range(-75,75,30)
p='multiPie_unmirror'
pm='multiPie_mirror'
labels=['Original Dataset separate','Original Dataset Aggregate','Mirrored Dataset separate','Mirrored Dataset aggregate']
for b in bins:
    #plt.figure()
    c=[pickle.load(open(p+'{}'.format(b))),pickle.load(open(p+'_aggregate{}'.format(b))),pickle.load(open(pm+'{}'.format(b))),pickle.load(open(pm+'_aggregate{}'.format(b)))]
    PyRoc.plot_multiple_roc(c,labels=labels,plot_average=False)
    dat=c[0].data
    plt.xlim((-0.01,1.0))
    plt.title('Pose: {}$^\circ$ - {}$^\circ$, Total: {}, n Open: {}'.format(b,b+30,len(dat),[i[0] for i in dat].count(1)))
    #plt.savefig('/home/attale00/Dropbox/tex_thesis/images/pose/mp_test_{}.pdf'.format(b))
    
    
#####################################
bins = range(-60,60,40)
for b in bins:
    #plt.figure()
    c=[pickle.load(open(p+'{}'.format(b))),pickle.load(open(p+'_aggregate{}'.format(b))),pickle.load(open(pm+'{}'.format(b))),pickle.load(open(pm+'_aggregate{}'.format(b)))]
    PyRoc.plot_multiple_roc(c,labels=labels,plot_average=False)
    dat=c[0].data
    plt.xlim((-0.01,1.0))
    plt.title('Pose: {}$^\circ$ - {}$^\circ$, Total: {}, n Open: {}'.format(b,b+40,len(dat),[i[0] for i in dat].count(1)))
    #plt.savefig('/home/attale00/Dropbox/tex_thesis/images/pose/mp_test_{}.pdf'.format(b))
    
################
bins = range(-60,60,40)

datalist = []
for b in bins:
    #plt.figure()
    r=pickle.load(open(pm+'{}'.format(b)))

    datalist.extend(r.data)
p_big_mirror = PyRoc.ROCData(datalist,linestyle='b-')

datalist = []
for b in bins:
    #plt.figure()
    r=pickle.load(open(p+'{}'.format(b)))

    datalist.extend(r.data)

p_big_unmirror = PyRoc.ROCData(datalist,linestyle='b-')


bins = range(-75,75,30)

datalist = []
for b in bins:
    #plt.figure()
    r=pickle.load(open(pm+'{}'.format(b)))

    datalist.extend(r.data)
p_small_mirror = PyRoc.ROCData(datalist,linestyle='b-')

datalist = []
for b in bins:
    #plt.figure()
    r=pickle.load(open(p+'{}'.format(b)))

    datalist.extend(r.data)

p_small_unmirror = PyRoc.ROCData(datalist,linestyle='b-')


orig = pickle.load(open('/home/attale00/fc_attributes/thesis_figures/patches/mp_combined'))
orig.linestyle='r-'
label = ['40 M','40 U','30 M','30 U','Combined']
PyRoc.plot_multiple_roc([p_big_mirror,p_big_unmirror,p_small_mirror,p_small_unmirror,orig],labels = label, plot_average = False)
plt.savefig('/home/attale00/Desktop/clf_nozoom.pdf')
plt.xlim((0.0,0.3))
plt.ylim((0.7,1.0))
plt.savefig('/home/attale00/Desktop/clf_zoom.pdf')

plt.show()
