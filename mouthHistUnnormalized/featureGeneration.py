from cv2 import imread
import numpy as np

class dataContainer:
    def __init__(self):
        self.data=[]
        self.target=[]
        self.fileNames=[]
        self.targetNum=[]
        self.classifiedAs=[]
        
    def splitInTestAndTraining(self, frac = .6):
        outTest = dataContainer()
        outTraining = dataContainer()
        stopInd = int(frac*len(self.data))
        idx = np.arange(0,len(self.data))
        np.random.shuffle(idx)
        for i in xrange(0,stopInd):
            outTraining.data.append(self.data[idx[i]])
            outTraining.target.append(self.target[idx[i]])
            outTraining.fileNames.append(self.fileNames[idx[i]])
            outTraining.targetNum.append(self.targetNum[idx[i]])
            
        for i in xrange(stopInd,len(self.data)):
            outTest.data.append(self.data[idx[i]])
            outTest.target.append(self.target[idx[i]])
            outTest.fileNames.append(self.fileNames[idx[i]])
            outTest.targetNum.append(self.targetNum[idx[i]])
        return (outTraining,outTest)
        
        
        

def getHistogram(nbins,roi,labelFileDict=None,hrange=(1.0,255,0),path=None,ending=None):
    '''
    nbins: number of bins for each histogram
    roi: tuple with the content (rmin,rmax,cmin,cmax)
    labelFileDict: dictionary containing filename-label pairs, None for unlabeled data
    path: path to folder containing images
    ending: optional alternative ending
    
    '''
    #get grayscale file
    out = dataContainer()
    
    k = labelFileDict.keys();
    
    
    for f in k:
        prefix = f.split('.')[0]
        f_name = path+prefix+ending
        im = imread(f_name,-1)
  
        ex = im[roi[0]:roi[1],roi[2]:roi[3]]

        vals,bins = np.histogram(ex,bins=nbins,range=hrange)
        label = labelFileDict[f]
        out.data.append(vals)
        out.target.append(label)
        out.fileNames.append(f)

    return out
