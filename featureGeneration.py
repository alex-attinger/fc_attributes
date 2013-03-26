from cv2 import imread
import numpy as np

def getHistogram(nbins,roi,labelFileDict=None,path=None,ending=None):
    '''
    nbins: number of bins for each histogram
    roi: tuple with the content (rmin,rmax,cmin,cmax)
    labelFileDict: dictionary containing filename-label pairs, None for unlabeled data
    path: path to folder containing images
    ending: optional alternative ending
    
    '''
    #get grayscale file
    data = []
    target = []
    
    k = labelFileDict.keys();
    
    
    for f in k:
        prefix = f.split('.')[0]
        f_name = path+prefix+ending
        im = imread(f_name)
        ex = im[roi[0]:roi[1],roi[2]:roi[3]]
        vals,bins = np.histogram(ex,bins=10,range=(1,255))
        label = labelFileDict[f]
        data.append(vals)
        target.append(label)

    return (data,target)
