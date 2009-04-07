#!/usr/bin/python
#coding:utf-8

import numpy as N
from mvpa.datasets.channel import ChannelDataset

from mvpa.measures.base import FeaturewiseDatasetMeasure

"""Module providing tools for datasets based on discrete events, like spike
recordings.

work in progress
"""

class SpikeDataset(ChannelDataset):
    """Acts like a ChannelDataset, but assumed to have discrete units"""
    #Hmm... I moved all improvements up to ChannelDataset, so far everything else
    # seems to fit better in a specific implementation, but a case could be made
    # to abstract the implementations more if the backends (ie plexon) can provide
    # a generic interface.  Since I only work with Plexon, I will leave this
    # issue alone for now...
    
    # NB: some usage in this module require changes to ChannelDataset, ie 
    # self.lowpass, self.selectChannels, self.getTime() etc
    
    
class PlexonSpike(SpikeDataset):
    def __init__(self,samples=None, plxInitArgs={}, sliceMethod='word', sliceArgs={}, 
                             sampleArgs={}, std=0, dsattr={}, **kwargs):
        """Init a SpikeDataset from a plexon file
        
        samples: one of:
        Plexon object
        path to one (loaded with plxInitArgs)
        
        samples will be created from the trials in the plx object, but other
        dataset attributes should be set through SpikeDsetArgs
        
        To create the dataset, the spikes must be sliced from a timeseries into
        samples.  This is done either with Plexon.sliceByTime or
        Plexon.sliceByWord, so sliceMethod should be set to 'word' or 'time'
        
        sliceArgs are then sent to the slice function, and the samples matrix
        is built
        
        regenerateSamples is called with sampleArgs to convert a plexon spike
        train into a ChannelDataset
        
        A few goodies are stored in dsattr (ie self._dsattr):
        
        plx (the Plexon object)
        slicedSpikes
        slicedChannels # Same as slicedSpikes but in a linear order
        slicedWords # The strobed words
        slicedEvents
        
        std: smooths data via lowpass gaussian filter
        """
        
        
        from mvpa.misc.io import plexon
        if isinstance(samples, plexon.Plexon):
            dsattr['plx'] = samples
        elif isinstance(samples, str):
            dsattr['plx'] = plexon.Plexon(samples, **plxInitArgs)
        if not isinstance(samples, N.ndarray) and 'plx' in dsattr.keys() and not 'data' in kwargs.keys():
            sliceMethod = sliceMethod.lower()
            if sliceMethod == 'word':
                (spks, evts, words) = dsattr['plx'].sliceByWord(**sliceArgs)
            elif sliceMethod == 'time':
                (spks, evts, words) = dsattr['plx'].sliceByTime(**sliceArgs)
            else:
                raise ValueError('Unknown slicing option: ' + sliceMethod)
            
            (dsattr['slicedSpikes'], dsattr['slicedEvents'], dsattr['slicedWords']) = (spks, evts, words)
            
            (samples, sampleprops) = self.generateSamples(spikes=spks, **sampleArgs)
            dsattr['slicedChannels'] = sampleprops['slicedChannels']
            del sampleprops['slicedChannels']
            if not 'labels' in kwargs.keys():
                kwargs['labels'] = N.arange(samples.shape[0])
            kwargs.update(**sampleprops)
        SpikeDataset.__init__(self, samples=samples, dsattr=dsattr, **kwargs)
        if std:
            self.lowpass(std)
        pass
    def regenerateSamples(self, std=0):
        (samples, props) = self.generateSamples(dt=self.dt, tStart=self.t0,
                                                tEnd = self.t0+(1+self.ntimepoints)*self.dt)
        self.samples = self.mapForward(samples)
        if std:
            self.lowpass(std)
    def generateSamples(self, dt=0.001, tStart=None, tEnd=None, spikes=None):
        """Creates the samples matrix from sliced spikes
        
        spikes: an appropriately formed dict, or if None, uses own copy
        #sigma: std of a Gaussian to smooth the data (in dt units). 0 -> no smooth
        dt: histogram bin width (seconds)
        tStart: time to start histogram (defaults: min)
        tEnd: time to stop histogram (defaults: max)
        
        dictOfProps contains t0, dt, channelids and slicedChannels if spikes
        were sent (otherwise it's stored locally)
        
        returns:
        (samples (nsamples * nchannels * ntimepoints), dictOfProps)
        (can't directly set these, since this function is called during init, 
        but you can probably directly set them postinit)
        """
        if spikes is None:
            spikes = self._dsattr['slicedSpikes']
            returnChannels=False
        else:
            returnChannels=True
        # Set the parameters
        s = []
        ids = []
        channels=[]
        setStart = False
        setEnd = False
        if tStart is None:
            tStart = float('inf')
            setStart = True
        if tEnd is None:
            tEnd = float('-inf')
            setEnd = True
        if setStart or setEnd:
            for c in spikes.values():
                for u in c.values():
                    if setStart:
                        tStart = min(tStart, min([t[0] for t in u if len(t)]))
                    if setEnd:
                        tEnd = max(tEnd, max([t[-1] for t in u if len(t)]))
        
        # Generates the data!!
        #if sigma:
            #dtype=float
        #else:
        dtype=float
        bins = N.arange(tStart, tEnd, dt)
        for cid,c in spikes.items():
            for  uid,u in c.items():
                ids.append('%i.%02i'%(cid, uid))
                channels.append(u)
                s.append(N.reshape(
                    N.asarray([N.histogram(trial, bins=bins, new=True)[0] for trial in u], dtype=dtype),
                (len(u), 1, bins.shape[0]-1)))
                
        samples = N.concatenate(s, axis=1)
        dictOfProps = {'t0':tStart, 'dt':dt, 'channelids':ids}
        if returnChannels:
            dictOfProps['slicedChannels'] = channels
        else:
            self._dsattr['slicedChannels'] = channels
        return (samples, dictOfProps)

    
class PoissonMu(FeaturewiseDatasetMeasure):
    def _call(self, dset):
        """Calculates mu (in Hz if samplingrate is available)"""
        try:
            mult = dset.samplingrate
        except AttributeError:
            mult = 1
        return mult*dset.samples.mean(axis=0)
class PoissonSigDiff(FeaturewiseDatasetMeasure):
    def __init__(self, splitter=None, maxDiff=100,**kwargs):
        """Object which runs a signifance test featurewise across a dataset
        
        The pvalue of the mean of one class being greater than the other is 
        returned.  A Skellam distribution is used to calculate significance
        (which is the difference of Poissons)
        
        A splitter must be provided yielding exactly 1 split of the data, which 
        will be the two sets compared
        
        Alternatively, you may call self.poissonDiffSignificance with two unique
        datasets, but this bypasses the standard class stuff (though I don't
        know if there's anything wrong with that)
        
        maxDiff: we must explicitly calculate the CMF, ie no closed form solution
        so, this is the range of possible differences we expect
        """
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)
        self.__PoissonMu = PoissonMu()
        self.__Splitter=splitter
        from numpy import arange
        self.possibles = arange(-maxDiff, maxDiff) # Viable differences in spike count
    def _call(self, dset):
        split = [s for s in self.__Splitter(dset)]
        return self.poissonDiffSignificance(*split[0])
    @staticmethod
    def skellamPMF(k, mu1, mu2=None): 
        if mu2 is None:
            mu2=mu1
        from scipy.special import iv # the modified besel function (i hope!!)

        return iv(k, 2.*N.sqrt(mu1*mu2))*N.exp(-(mu1+mu2))*(mu1/float(mu2))**(k/2.)
    
    def poissonDiffSignificance(self, dset1, dset2):
        """Returns the significance of each (channel x ntimepoints) being different
        under a Poisson distribution"""
        
        mu1 = self.__PoissonMu(dset1)
        n1 = dset1.nsamples
        mu2 = self.__PoissonMu(dset2)
        n2 = dset2.nsamples
        
        try:
            assert mu1.shape == mu2.shape
        except AssertionError:
            raise RuntimeError('Cannot compare datasets with different numbers of channels or timepoints')
    
        from scipy import interp
        muNull = (n1*mu1+n2*mu2)/(n1+n2)
        k = (mu2-mu1).ravel()
        myCmf = N.zeros(mu1.shape).ravel()
        for i in range(myCmf.size):
            cmf = PoissonSigDiff.skellamPMF(self.possibles, muNull.ravel()[i]).cumsum()
            cmf /= cmf[-1]
            myCmf[i] = interp(k[i], self.possibles, cmf)
        
        return 0.5 - abs(myCmf.reshape(mu1.shape)-0.5)

class PoissonCI(FeaturewiseDatasetMeasure):
    def __init__(self, alpha=0.05, **kwargs):
        """Returns the confidence bound assuming a Poisson distribution
        alpha - level of confidence (ie .05 yields the 95% CI)
        
        Since the bounds are symmetric for a Poisson, this returns (ub-mu)
        So, (ub, lb) = (mu+bound, mu-bound)
        mu is easily calculated with PoissonMu, or by simply averaging
        
        If the dataset defines samplingrate, it is calculated in Hz (implemented
        in PoissonMu)
        """
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)
        self.alpha = alpha
        self.mu = PoissonMu()
        
    def _call(self, dset):
        from scipy.stats import poisson, gamma
        mu = self.mu(dset)
        try:
            mult=dset.samplingrate
        except AttributeError:
            mult=1.
    
        return poisson.isf(self.alpha/2, mu)-mu
    def getMu(self, dset):
        return self.mu(dset)
    def getBounds(self, dset):
        """Returns (LB, UB)"""
        mu = self.mu(dset)
        b = self(dset)
        return (mu-b, mu+b)




def plotRaster(dset, channels=None, newFig=True, filt=True):
    import pylab
    for c in filter(lambda i: (channels is None) or i in channels or i in dset.channelids,
                    range(dset.O.shape[1])):
        if newFig:
            pylab.figure()
        pylab.title(dset.channelids[c])
        pylab.hold(True)
        if filt:
            tbeg=dset.t0
            tend=(dset.t0+(1+dset.ntimepoints)*dset.dt)
            def f(s):
                return s>=tbeg and s <=tend
        for (t,trial) in enumerate(dset._dsattr['slicedChannels'][c]):
            if filt:
                trial = filter(f, trial)
            if len(trial) > 0:
                pylab.scatter(trial, [t]*len(trial), s=0.5)
        
def plotRate(dset, channels=None, color='r', newFig=True):
    import pylab
    
    t=dset.getTime()
    p = dset.mapReverse(PoissonMu()(dset))
    (lb95, ub95) = map(dset.mapReverse, PoissonCI().getBounds(dset))
    (lb66, ub66) = map(dset.mapReverse, PoissonCI(1/3.).getBounds(dset))
    
    if not newFig:
        pylab.hold(True)
    for c in filter(lambda i: (channels is None) or i in channels or i in dset.channelids,
                    range(p.shape[0])):
        if newFig:
            pylab.figure()
            pylab.title(dset.channelids[c])
            pylab.xlabel('PST')
            pylab.ylabel('Hz')
            pylab.hold(True)
        (x,y) = pylab.poly_between(t, lb95[c,:],ub95[c,:])
        pylab.fill(x,y, color, alpha=0.25)
        (x,y) = pylab.poly_between(t, lb66[c,:], ub66[c,:])
        pylab.fill(x,y, color, alpha=0.25)
    
        pylab.plot(t, p[c,:], color)
        if newFig:
            pylab.axhline(0, color='k')
            pylab.axvline(0, color='k')
    pylab.hold(False)

def loadTestPlx():
    """Loads one of two sample plexon files from the mvpa data dir"""
    from mvpa.misc.io import plexon
    return PlexonSpike(plexon.loadTestPlx(), sliceMethod='time', std=10,
                       sliceArgs={'times':[(a, a+1) for a in range(5)], 'ignore0':False})

def loadTestRealDataset():
    """This won't work anywhere except Scott's computer...sorry"""
    plx = plexon.Plexon('/home/scott/data/physiology/riseComplexDetection/akebono/record/akebono_10_13_2008-01.plx')
    sliceMethod='word'
    sliceArgs={'ignore0':False, 'startWord':-.5, 'zeroWord':1001, 'endWord':2, 'limAsTime':True}
    return PlexonSpike(plx, sliceMethod=sliceMethod, std=10,
                       sliceArgs=sliceArgs)
    
if __name__ == '__main__':
    import pylab
    from mvpa.datasets.splitters import HalfSplitter
    p = loadTestPlx()
    splitter = HalfSplitter()
    (p1, p2) = splitter(p).next()
    sig = PoissonSigDiff(splitter)
    pvals = p.mapReverse(sig(p))
    #p2.samples = p2.samples[-1:-1001:-1, :]
    pylab.figure()
    for c in range(4):
        pylab.subplot(2,2,c+1)
        plotRaster(p.selectChannels(c), newFig=False)
    plotRate(p1.selectChannels(3))
    plotRate(p2.selectChannels(3),newFig=False, color='b')
    #sig = poissonDiffSignificance(p1, p2)

    pylab.figure()
    pylab.plot(p.getTime(), pvals[-1,:])
    pylab.show()
    
