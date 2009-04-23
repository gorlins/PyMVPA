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
        bins = N.arange(tStart, tEnd, dt)
        for cid,c in spikes.items():
            for  uid,u in c.items():
                ids.append('%i.%02i'%(cid, uid))
                channels.append(u)
                s.append(N.reshape(
                    N.asarray([N.histogram(trial, bins=bins, new=True)[0] for trial in u], dtype=float),
                (len(u), 1, bins.shape[0]-1)))
                
        samples = N.concatenate(s, axis=1)
        dictOfProps = {'t0':tStart, 'dt':dt, 'channelids':ids}
        if returnChannels:
            dictOfProps['slicedChannels'] = channels
        else:
            self._dsattr['slicedChannels'] = channels
        return (samples, dictOfProps)

    
class PoissonMu(FeaturewiseDatasetMeasure):
    def __init__(self, mult=True, **kwargs):
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)
        self.mult=mult
    def _call(self, dset):
        """Calculates mu (in Hz if samplingrate is available)"""
        if self.mult:
            try:
                mult = dset.samplingrate
            except AttributeError:
                mult = 1
        else:
            mult=1
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
        """Returns the significance of each feature being different
        under a Poisson distribution"""
        
        mu1 = self.__PoissonMu(dset1)
        n1 = dset1.nsamples
        mu2 = self.__PoissonMu(dset2)
        n2 = dset2.nsamples
        
        try:
            assert mu1.shape == mu2.shape
        except AssertionError:
            raise RuntimeError('Cannot compare datasets with different numbers of features')
    
        from scipy import interp
        muNull = (n1*mu1+n2*mu2)/(n1+n2)
        k = (mu2-mu1).ravel()
        myCmf = N.zeros(mu1.shape).ravel()
        for i in range(myCmf.size):
            cmf = PoissonSigDiff.skellamPMF(self.possibles, muNull.ravel()[i]).cumsum()
            cmf /= cmf[-1]
            myCmf[i] = interp(k[i], self.possibles, cmf)
        
        return 0.5 - abs(myCmf.reshape(mu1.shape)-0.5)

class PoissonCI(object):
    def __init__(self, alpha=0.05, **kwargs):
        """Wraps PoissonBound to deal with upper, lower, and mu"""
        self.ub = PoissonBound(alpha=alpha, upper=True, **kwargs)
        self.lb = PoissonBound(alpha=alpha, upper=False, **kwargs)
        self.mu = PoissonMu()
        self.ub.mu = self.mu
        self.lb.mu = self.mu
        self.alpha = alpha
    @staticmethod
    def mle(samples, mn=0.01):
        """Returns (khat, thetahat) MLE estimates of a gamma distribution for 
        samples across the first dimension 
        uses approximation found here:
        http://en.wikipedia.org/wiki/Gamma_distribution#Parameter_estimation
        
        :Parameters:
          samples: (samples x features etc) Numpy array
          mn: forced minimum of each sample (requires log, so can't be 0)
        """
        mu = samples.mean(axis=0)
        samples = N.maximum(samples, mn)
        s = N.log(mu) - N.log(samples).mean(axis=0)
        khat = (3-s+N.sqrt((s-3)**2+24*s))/(12*s)
        thetahat = mu/khat
        return (khat, thetahat)
    @staticmethod
    def llBetaMu(samples, beta, mu):
        n = samples.shape[0]
        t1 = N.log(samples).sum(axis=0)
        t2 = samples.sum(axis=0)
        from scipy import special
        return -n*special.gammaln(beta) + n*beta*N.log(beta/mu) + beta*(t1 -t2/mu) - t1
    
    @staticmethod
    def calcR(samples, betahat, muhat, mu):
        (beta, theta) = PoissonCI.mleGivenMu(samples, mu)
        R = N.sign(muhat-mu)*N.sqrt(2*(PoissonCI.llBetaMu(samples, betahat, muhat) - PoissonCI.llBetaMu(samples, beta, mu)))
        return (R, beta)
        
    @staticmethod
    def calcStat(samples, mu):
        (betahat, thetahat) = PoissonCI.mle(samples)
        muhat=betahat/thetahat
        (R, beta) = PoissonCI.calcR(samples, betahat, muhat, mu)
        Q = PoissonCI.calcQ(samples.shape[0], beta, mu, betahat, muhat)
        p = R - R**-1*N.log(R/Q)
        return (p, Q, R)
    @staticmethod
    def calcQ(n, beta, mu, betahat, muhat):
        from scipy.special import polygamma
        return N.sqrt(n*betahat)*(muhat/mu - 1)*(polygamma(1,betahat)-betahat**-1)**0.5 * (polygamma(1,beta)-beta**-1)**-0.5
    
    @staticmethod
    def mleGivenMu(samples, mu):
        """Calculates the MLE under the assumption of a given mean"""
        n=samples.shape[0]
        s = N.log(mu) + samples.sum(axis=0)/(n*mu) - N.log(samples).mean(axis=0)-1
        khat = (3-s+N.sqrt((s-3)**2+24*s))/(12*s)
        thetahat = mu/khat
        #assert(N.all(ahat*thetahat == mu))
        return (khat, thetahat)
    def getBounds(self, dset):
        """Gets the confidence intervals
        
        :Returns:
          (lb, mu, ub) which are all FeaturewiseDatasetMeasure results
        """
        from scipy import special
        try:
            mult = dset.samplingrate
        except AttributeError:
            mult = 1
        s = dset.samples*mult
        scale = s.mean()
        s = s/scale
        # MLE
        (ahat, bhat) = self.mle(s)
        from scipy.stats import norm, gamma
        mu = ahat*bhat*scale
        bhat*=scale
        X = norm.isf(self.alpha/2)
        pm = X*mu*N.sqrt(X**2 + bhat**-2*(1/(mu**2) - 1))
        
        #z = norm.isf(self.alpha/2)
        #std = N.sqrt(ahat)*bhat*scale
        #lb = N.maximum(0,mu-z*std)
        #ub = mu+z*std
        #lb = gamma.isf(1-self.alpha/2, ahat, scale=bhat*scale)
        #ub = gamma.isf(self.alpha/2, ahat, scale=bhat*scale)
        
        # Compute CIs on the log scale for both params
        #z = s/bhat
        #L = (ahat - 1) *N.log(z)-z - special.gammaln(ahat) - N.log(bhat)

        ##% Sum up the individual contributions, and return the negative log-likelihood.
        #nlogL = -L.sum(axis=0)

        ##% Compute the negative hessian at the parameter values, and invert to get
        ##% the observed information matrix.
        #dL11 = -special.polygamma(1, ahat)#-repmat(special.polygamma(1,ahat),size(z));
        #dL12 = -1./bhat#-repmat(1./bhat,size(z));
        #dL22 = -(2.*z-ahat) / (bhat**2);
    
        #nH11 = -dL11*dset.nsamples#-dL11.sum(axis=0)
        #nH12 = -dL12*dset.nsamples#-dL12.sum(axis=0)
        #nH22 = -dL22.sum(axis=0)
        ##avar = N.matrix([[nH11, nH12],[nH12, nH22]]).I
        ## Do it manually
        #det=(nH11*nH22 - nH12**2)
        ##acov = N.asarray([[nH22, -nH12], [-nH12, nH11]])/det

        #from scipy.stats import gamma, norm
        ##logL = N.log(gamma.pdf(mult*dset.samples, ahat, scale=bhat)).sum(axis=0)
        ##[logL, acov] = gamlike(parmhat, x, censoring, freq);
        ##selog = N.sqrt(N.asarray([nH22/(ahat**2), nH11/(bhat**2)])/det)
        #seloga = N.sqrt(nH22/det)/ahat
        #selogb = N.sqrt(nH11/det)/bhat
        #lahat = N.log(ahat)
        #lbhat = N.log(bhat)
        ##a = self.alpha
        ##self.alpha = a**(1/2.)
        ##partialalpha = N.sqrt(self.alpha)
        #asteps = 1-N.arange(.01, 1, .01)
        #agrid = asteps.reshape((asteps.size, 1)).repeat(lahat.size, axis=1)
        #lagrid=lahat.reshape((1, lahat.size)).repeat(asteps.size, axis=0)
        #lbgrid=lbhat.reshape((1, lbhat.size)).repeat(asteps.size, axis=0)
        #segrida=seloga.reshape((1, lbhat.size)).repeat(asteps.size, axis=0)
        #segridb=selogb.reshape((1, lbhat.size)).repeat(asteps.size, axis=0)
        
        #isfa=N.exp(norm.isf(agrid, loc=lagrid, scale=segrida))
        #isfb=N.exp(norm.isf(agrid, loc=lbgrid, scale=segridb))
        #mugrid = scale* isfa.reshape((asteps.size, 1, dset.nfeatures)).repeat(asteps.size, axis=1) * isfb.reshape((1, asteps.size, dset.nfeatures)).repeat(asteps.size, axis=0)
        #mugrid=mugrid.reshape((99**2, dset.nfeatures))
        ##mugrid = isfa*isfb*scale
        ##mugrid.sort(axis=0)
        #lb=mugrid[5,:]
        #ub=mugrid[-5,:]
        
        #uba = N.exp(norm.isf(partialalpha/2, loc=lahat, scale=seloga))
        #lba = N.exp(norm.isf(1-partialalpha/2, loc=lahat, scale=seloga))
        #ubb = N.exp(norm.isf(partialalpha/2, loc=lbhat, scale=selogb))
        #lbb = N.exp(norm.isf(1-partialalpha/2, loc=lbhat, scale=selogb))
        #self.alpha = a
        #lbb*=scale
        #ubb*=scale
        #lb=lba*lbb
        #ub = uba*ubb
        return (lb, mu, ub)
        
class PoissonBound(FeaturewiseDatasetMeasure):
    def __init__(self, alpha=0.05, priorA=0., priorB=0., upper=True, **kwargs):
        """Returns the confidence bound assuming a Poisson distribution
        
        :Parameters:
          alpha: level of confidence (ie .05 yields the 95% CI)
          
          upper: if True, returns upper bound, otherwise returns lower bound
          
          priorA: Prior distribution shape parameter a for the gamma distribution
            0 means uniform prior
            
          priorB: Scale (beta) parameter for the gamma.  0 is uniform.  Note that
            this is in terms of beta in the numerator for the gamma func, so that
            it acts as conjugate posterior to the Poisson.  This contrasts from 
            scipy/Matlab where beta is the denominator, so the inverse is taken
            internally
        
        :Returns:
          Upper (or lower) bound of 100*(1-alpha)% confidence interval
          
        Since the bounds are roughly symmetric for a Poisson, this only needs to
        be called once with upper, and the bounds can be expressed as:
        (lb, ub) = (mu-(ub-mu), ub)
        
        Otherwise, use two instances to calculate (lb, ub) exactly
        
        mu is easily calculated with PoissonMu, or by simply averaging
        
        If the dataset defines samplingrate, it is calculated in Hz (implemented
        in PoissonMu)
        
        """
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)
        self.alpha = alpha
        self.mu = PoissonMu()
        self.upper=upper
        self.priorA=priorA
        self.priorB=priorB
        
    def _call(self, dset):
        from scipy.stats import gamma
        try:
            s = dset.samplingrate*dset.samples
        except AttributeError:
            s = dset.samples
        
        #mu = s.mean(axis=0)
        #var = s.var(axis=0)
        #T = dset.nsamples*mu
        a = self.alpha/2.
        if not self.upper:
            a = 1-a
        #return gamma.isf(a, T+self.priorA,scale=1./(self.priorB+dset.nsamples))
        
        #return gamma.isf(a, mu**2/var, scale=var/mu)
        #h = map(gamma.fit, s.T)
        #return N.asarray(map(self.getISF, [a]*dset.nfeatures, h))
        (ahat, bhat) = self.mle(s)
        return gamma.isf(a, ahat, scale=bhat)
    @staticmethod
    def mle(samples, mn=0.01):
        """Returns (ahat, bhat) MLE estimates of a gamma distribution for samples
        across the first dimension 
        uses approximation found here:
        http://en.wikipedia.org/wiki/Gamma_distribution#Parameter_estimation
        
        :Parameters:
          samples: (samples x features etc) Numpy array
          mn: forced minimum of each sample (requires log, so can't be 0)
        """
        samples = N.maximum(samples, mn)
        s = N.log(samples.mean(axis=0)) - N.log(samples).mean(axis=0)
        ahat = (3-s+N.sqrt((s-3)**2+24*s))/(12*s)
        bhat = samples.mean(axis=0)/ahat
        return (ahat, bhat)
    @staticmethod
    def getISF(a, h):
        from scipy.stats import gamma
        return gamma.isf(a, h[-2], scale=h[-1])
    def getPosteriorCDF(self, x, mu, n):
        from scipy.stats import gamma
        return 1-gamma.sf(x, self.priorA+mu*n, scale=1./(self.priorB+n))
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
    (lb95, p, ub95) = map(dset.mapReverse, PoissonCI().getBounds(dset))
    (lb66, junk, ub66) = map(dset.mapReverse, PoissonCI(1/3.).getBounds(dset))
    opacity=.15
    for c in filter(lambda i: (channels is None) or i in channels or i in dset.channelids,
                    range(p.shape[0])):
        if newFig:
            pylab.figure()
        pylab.title(dset.channelids[c])
        pylab.xlabel('PST')
        pylab.ylabel('Hz')
        pylab.hold(True)
        (x,y) = pylab.poly_between(t, lb95[c,:],ub95[c,:])
        pylab.fill(x,y, color, alpha=opacity)
        (x,y) = pylab.poly_between(t, lb66[c,:], ub66[c,:])
        pylab.fill(x,y, color, alpha=opacity)
    
        pylab.plot(t, p[c,:], color)
        pylab.axhline(0, color='k')
        pylab.axvline(0, color='k')
    pylab.hold(False)

def loadTestPlx(**loadargs):
    """Loads one of two sample plexon files from the mvpa data dir"""
    from mvpa.misc.io import plexon
    return PlexonSpike(plexon.loadTestPlx(), sliceMethod='time',
                       sliceArgs={'times':[(a, a+1) for a in range(5)], 'ignore0':False}, **loadargs)

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
    p = loadTestPlx(std=20)
    splitter = HalfSplitter()
    (p1, p2) = splitter(p).next()
    #sig = PoissonSigDiff(splitter)
    #pvals = p.mapReverse(sig(p))
    #p2.samples = p2.samples[-1:-1001:-1, :]
    pylab.figure()
    for c in range(4):
        pylab.subplot(2,2,c+1)
        plotRaster(p.selectChannels(c), newFig=False)
    pylab.figure()
    for c in range(4):
        pylab.subplot(2,2,c+1)
        plotRate(p.selectChannels(c), newFig=False)
        
    plotRate(p1.selectChannels(3))
    plotRate(p2.selectChannels(3),newFig=False, color='b')
    #sig = poissonDiffSignificance(p1, p2)
    #plotRate(p)
    #pylab.figure()
    #pylab.plot(p.getTime(), pvals[-1,:])
    pylab.show()
    
