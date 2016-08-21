import numpy as np
from matplotlib import pyplot as plt

class ChangeDetector():
    
    def __init__(self, x, blocksize, word='', cutoff=3, poisson=False):
        ''' Mostly this is implemented where x is sparse.'''
        self.x = x
        self.blocksize = blocksize
        self.word = word
        self.cutoff = cutoff
        self.poisson = poisson
        self.index = np.arange(0, self.x.shape[0], blocksize, dtype=np.int64)
        
        self.moving_average, self.index_moving_average = moving_average(self.x, self.blocksize)
        
        self.error = 0.

        
    def iterate(self, display=True, showProgress=False):
        self.mean, self.std, self.anomaly, self.error = calculateAnomalies(self.x, self.index, cutoff=self.cutoff, poisson=self.poisson)
        
        abort = False
        i = 0
        while not(abort):
            if (display and showProgress):
                plt.figure()
                self.plot()
                # plt.plot(self.index_moving_average, self.moving_average, 'k-')
                # plotAnomaly(self.mean, self.std, self.anomaly, self.index, word)

            subindex, abort = annealAnomalies(self.anomaly)
            
            
            
            if abort:
                break
            
            self.index = self.index[subindex]
            
            self.mean, self.std, self.anomaly, self.error = calculateAnomalies(self.x, self.index, cutoff=self.cutoff, poisson=self.poisson)

            i += 1
            if i > 100:
                assert('Too many iterations!!!')
                break
        if (display and not(showProgress)):
            self.plot()
                
    def plot(self):
        plt.plot(self.index_moving_average, self.moving_average, 'k-')
        plotAnomaly(self.mean, self.std, self.anomaly, self.index, self.word)
        # return (mean, std, anomaly)

    


def moving_average(x, blocksize):
    cumsum = np.cumsum(x.toarray()) 
    moving_average = (cumsum[blocksize:] - cumsum[:-blocksize]) / (1.*blocksize)
    index_moving_average = (0.5*blocksize) + np.arange(len(moving_average))
    
    return (moving_average, index_moving_average)


def plotAnomaly(mean, std, anomaly, index, word=''):
    for i in range(len(index)-1):
        if (i==0 or anomaly[i]):
            color = 'b'
        else:
            color = 'r'
        plt.fill_between([index[i], index[i+1]],
                         [mean[i]-std[i], mean[i]-std[i]],
                         [mean[i]+std[i], mean[i]+std[i]],
                         facecolor=color,
                         edgecolor='none',
                         alpha=0.3)
        plt.plot([index[i], index[i+1]],
                 [mean[i], mean[i]], color=color,
                 lw=2)
        plt.title(word)

def calculateAnomalies(x, index, cutoff=2, poisson=False):
    numBlocks = len(index)-1
    mean = np.zeros((numBlocks,))
    std = np.zeros((numBlocks,))
    anomaly = np.zeros((numBlocks,), dtype=np.bool_)
    error = 0.

    for i in range(numBlocks):
        blocksize = index[i+1] - index[i]
        currentBlock = x[index[i]:index[i+1]].toarray()
        
        mean[i] = np.mean(currentBlock)
        
        if poisson:
            std[i] = np.sqrt(1.*np.sum(currentBlock))/blocksize
        else:
            std[i] = np.std(currentBlock)/np.sqrt(blocksize)

        if i > 0:
            z = (np.abs(mean[i] - mean[i-1])) / np.sqrt( std[i]**2 + std[i-1]**2 )
            error += z
            if z > cutoff:
                anomaly[i] = True
            
        
    return (mean, std, anomaly, error)
    
def annealAnomalies(anomaly):
    i = 1
    index = [0]

    skippedLastValue = False
    
    abort = True
    
    while i < len(anomaly):
        if (anomaly[i] or skippedLastValue):
            index.append(i)
            skippedLastValue = False
        else:
            skippedLastValue = True
            abort = False
        i += 1

    index.append(i)

    return (index, abort)

