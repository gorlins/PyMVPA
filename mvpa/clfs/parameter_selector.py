# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""This module runs brute force parameter selection on any given classifier.

A great deal of work is needed to make it optimal for retrainable classifiers,
but currently, it should work for any cross-foldable dataset and classifier
(albeit slowly, though not unreasonably so)

Though functional, this code was written while I was first learning python,
so there are certainly areas that need cleaning up and/or rethinking.
Todo: rewrite on scipy.optimize backend, use ParameterSelection as an abstract
class and write inheritors implementing different optimization solvers.

Also plotting is slow and messy - cleanup!!

Probably should be reconciled with model_selector somehow...

work in progress
"""
import copy
import numpy as N
from time import time
from mvpa.base import externals
externals.exists("scipy", raiseException=True)

from scipy.ndimage import gaussian_filter
from mvpa.clfs.transerror import TransferError
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError


#from scipy.optimize import brute, fmin
class ParameterSelection(object):
    """Runs a grid search algorithm on a classifier and dataset"""
    def configureSubplots(self, nrows, ncols):
        """Resets the subplot figure count and shapes the subplot"""
        self._nrows=nrows
        self._ncols=ncols
        self._n=0
    def __init__(self, classifier, params, splitter, scales = None, 
                 defaults=None, cv=None, te=None, priorWeight=.1,
                 factors = 2., iterations=2, plot=True, nrows=1, ncols=1, 
                 manipulateClassifier=False, log=True):
        """Inits a ParameterSelector to do grid evaluation on a data cross-fold
        to find the best classifier parameters
        
        Slightly better than the default scipy grid selector in that it memoizes
        but the implementation could cerainly use a lot of work
        
        Use example: clf = SVM(...), dset=Dataset(...), splitter=NFoldSplitter(...)
        psel = ParameterSelection(clf, 'C', splitter, **kwargs)
        psel(dset)
        clf.train(dset), clf.classify(anotherdset)
        
        :Returns:
          Object which implements __call__ (or select()) which runs cross-
          validated parameter selection on the input classifier, leaving that
          classifier trained on the optimal parameters.  
        
        :Parameters:
          classifier: a mvpa clf instance
          
          params: tuple of strings indicating the parameters to select
          eg ('C', 'gamma')
          
          splitter: mvpa.datasets.splitter instance
          
          scales, defaults, factors: tuple of settings (matching parameter in 
          params) or a single setting which affects all.  
          
          scales: array (or tuple of such) defining the range in which to 
          evaluate the grid
          
          priorWeight: A prior is assigned during error selection, and this allows
          scaling.  The prior is the geometric distance (or dist in log space) from 
          the default parameter, scaled to [0 errorMax-errorMin]/current_iteration,
          and added to the error.  It can be further weighed here (ie set 0 -> no prior)
          
          log: indicates whether search (and plot) is done in logspace.  
          can be a tuple.
          
          if log==True, search is performed on the grid of scales arrays
          defaults*2**(scales+offset), unless if factor==1, the grid in scales 
          is directly evaluated.  Due to the implementation, log == True 
          allows for a search outside the original scale (due to the offset) 
          with each iteration.  Currently, if log==False, factor must = 1 since 
          grid is not reevaluated relative to the best position
        
          factor: reduce each grid by factor in each iteration - chose 1 for a 
          constant grid 
          
          plot: whether to plot the results (must be 1 or 2d)
          nrows, ncols: controls subplots (can be set with configureSubplots),
          should you want to do parameter selection on different chunks of data
          in one plot for convenience
          
          cv: CrossValidatedTransferError instance (created automatically if None)
          te: TransferError instance (created automatically if None)
          
          maninpulateClassifier: if True, alters the classifier's train method
          to automatically select the parameters.  Uses module function makePsel
        """
        if te is None:
            te = TransferError(classifier)
        
        if cv is None:
            cv = CrossValidatedTransferError(te, splitter)

        self._psel = cv
        self._priorWeight=priorWeight
        #self._psel.states.enable('confusion')
        #self._psel.states.enable('training_confusion')
        self._clf = classifier
        self._defaults = {}
        self._scales = {}
        self._log = {}
        self.configureSubplots(nrows,ncols)
        if not isinstance(params, tuple):
            params = tuple([params])
        self._plot = plot and len(params)<=2
        self._plot2d = len(params)==2
        self._params = params
        self._factors = {}
        self._iter = 1
        
        # Parses the inputs for each parameter
        for (i,param) in enumerate(params):
            # Scales
            if not scales is None:
                if isinstance(scales, tuple):
                    self._scales[param]=scales[i]
                else:
                    self._scales[param]=scales
            else:
                self._scales[param]=N.arange(-20., 21., 2., dtype=float)
                
            # Default (initial) values
            if not defaults is None:
                if isinstance(defaults,tuple):
                    self._defaults[param]=defaults[i]
                else:
                    self._defaults[param]=defaults
            else:
                self._defaults[param] = -1.
                                              
            # Factor
            if isinstance(factors, tuple):
                self._factors[param]=factors[i]
            else:
                self._factors[param]=factors
                
            # Only sets iterations if there is a real factor
            if not self._factors[param] == 1:
                self._iter = iterations
                
            # Log
            if isinstance(log, tuple):
                self._log[param]=log[i]
            else:
                self._log[param]=log
        if manipulateClassifier:
            makePsel(classifier, self)
    def __call__(self, dset, title=''):
        self.select(dset, title=title)

    def select(self, dset, title=''):
        """Runs the parameter selection on a given mvpa dataset, internally 
        setting classifier parameters
        
        if plotting, you can title the figure with the keyword title
        """
        # Init parameters
        t0=time()
        params = self._params
        scales = copy.deepcopy(self._scales)
        offsets = {}
        origins = {}
        self._n+=1
        for p in params:
            offsets[p] = 0
            if self._defaults[p] == -1.:
                try:
                    pupper = p[0].upper() + p[1:]
                    origins[p] = getattr(self._clf, '_getDefault'+pupper)(dset.samples)
                except AttributeError:
                    try:
                        origins[p] = getattr(self._clf, '_getDefault'+pupper)(dset)
                    except Exception:
                        origins[p] = 1.
            else:
                origins[p] = 1.
        # If can't run cross validation, set default parameters
        nsplits = len([1 for s in self._psel.splitter(dset)])
        if not nsplits > 1:
            Warning('No independent splits for CV parameter selection')
            for p in self._params:
                try:
                    setattr(self._clf, p, origins[p])
                except Error:
                    pass
            self.best = 0
            self.worst = 0
            self.time = 0
            return
                
        self._errs = errs = {}
        self.worst = 0.
        
        # Creates figure
        if self._plot:
            import pylab
            self._colormap = pylab.cm.jet
            from matplotlib.collections import RegularPolyCollection
            scatterargs={'edgecolors':'none', 'marker':'o'}#,'cmap':pylab.cm.Paired, 'vmin':0, 'vmax':1}
            cRGB = 1-N.asarray([[1.,1.,1.]])
            #shouldTurnOff = not pylab.isinteractive()
            #pylab.ion()
            px = params[0]
            if self._plot2d:
                py = params[1]
            
            self.__axes = pylab.subplot(self._nrows, self._ncols, self._n)
            axes = self.__axes
            axes._autoscaleon=True
            #if shouldTurnOff:
                #pylab.ioff()
                
            if self._plot2d:
                if self._log[px] and not self._log[py]:
                    axes.semilogx()
                elif self._log[py] and not self._log[px]:
                    axes.semilogy()
                elif self._log[px] and self._log[py]:
                    axes.loglog()
                else:
                    pass#pylab.Figure()
                pylab.ylabel(py)
            else:
                if self._log[px]:
                    axes.semilogx()
                else:
                    pass
                pylab.ylabel('CV Error')
            
            pylab.xlabel(px)
            pylab.title(title)
            init=True
            xarray = []
            yarray = []
            counter=0
            self.__scatteroffsets = []
            self.__facecolors=[]
            self.__collection = RegularPolyCollection(8, sizes=(50,), rotation=N.pi/8,
                                                      facecolors=self.__facecolors,
                                                      edgecolors='none',
                                                      offsets=self.__scatteroffsets,
                                                      transOffset=axes.transData
                                                      )
            #axes.add_collection(collection)
            pylab.draw()
     

        ###        
        # Runs the optimization
        ###
        
        # trying to use scipy opt, not yet working
        #self._errs={}
        self._dset = dset
        #from scipy.optimize import fmin, brute
        #if self._plot:
            #callback = self._plotX
        #else:
            #callback = None
        #origx = self._pToX([origins[p] for p in self._params])
        #optx = fmin_bfgs(self._evalP, origx, callback=callback)
        #self.best = min(self._errs.values())
        #self.worst = max(self._errs.values())
        
        # Brute force (custom gridding)
        err = N.zeros(tuple([len(s) for s in scales.itervalues()]))
        prior = err.copy()
        originp = []
        for p in params:
            if self._log[p]:
                originp.append(N.log(origins[p]))
            else:
                originp.append(origins[p])
        originp=N.asarray(originp).reshape([len(params)]+[1]*len(params))
        for iteration in range(self._iter):
            # Build grids
            grids = {}
            for p in params:
                if self._log[p]:
                    grids[p] = origins[p]*2.**(scales[p]+offsets[p])
                else:
                    grids[p] = scales[p]
            pind = N.indices([grids[p].size for p in params])
            pgrid=pind.copy().astype(float)
            for (i,p) in enumerate(params):
                pgrid[i] = grids[p][pind[i]]
                
            # Runs the actual selection in map
            paramvec = zip(*[p.ravel() for p in pgrid])
            err = N.asarray(map(self._evalP, paramvec)).reshape(pgrid[0].shape)
            priorgrid = pgrid.copy()
            for (i,p) in enumerate(params):
                if self._log[p]:
                    priorgrid[i] = N.log(priorgrid[i])
            prior = N.sqrt(N.power(priorgrid-originp, 2).sum(axis=0))
            
            errscale = N.max(err.max()-err.min(), .01)
            moderr = err+self._priorWeight*errscale*(prior-prior.min())/(prior.max()-prior.min())/(iteration+1)
            mins = moderr==moderr.min()
            if mins.sum() == 1:
                eid = moderr.argmin()
            else:
                sigma=1
                while mins.sum() > 1 and sigma<5:
                    g = gaussian_filter(moderr, sigma,cval=1., mode='constant')
                    mins = N.where(mins, g, g.max())==g.min()
                    sigma+=1
                eid = N.where(mins, g, g.max()).argmin()
            eidunravel = N.unravel_index(eid, err.shape)
            self.best = err.ravel()[eid]
            
            # Set best parameters, update offsets, rescales
            for (i, p) in enumerate(params):
                setattr(self._clf, p, pgrid[i].ravel()[eid]) # Sets to best!!
                if self._log[p]:
                    # Recenter scales on best point
                    offsets[p]+=scales[p][eidunravel[i]]
                    
                    # Correct for non-centered scales
                    offsets[p]-=scales[p][N.floor(len(scales[p])/2.)]/self._factors[p]
                    
                    # Rescale
                    scales[p]/=self._factors[p] 
            if self._plot:
                self.plotPoints(paramvec, [self._errs[p] for p in paramvec])
                pylab.draw()
            self.worst = max(self.worst, err.max())
        
        self.time=time()-t0
        if self._plot:
            pylab.draw()
            if self._plot2d:
                pylab.axhline(origins[py], color='k', linestyle='dotted')
                pylab.axhline(getattr(self._clf, py), color='k')
            else:
                pylab.axhline(errs[(getattr(self._clf,px),)], color='k')
            pylab.axvline(origins[px], color='k', linestyle='dotted')
            pylab.axvline(getattr(self._clf, px), color='k')
            pylab.draw()
        
        self._dset = None# dont want to hold onto it
    def _plotX(self, x):
        p = self._xToP(x)
        err = self._errs[p]
        xx = p[0]
        if self._plot2d:
            yy = p[1]
        else:
            yy = err
        self.plotPoint(xx,yy,err)
    def _pToX(self, p):
        x = []
        for (i,param) in enumerate(self._params):
            if self._log[param]:
                x.append(N.log(p[i]))
            else:
                x.append(p[i])
        return tuple(x)
    def _xToP(self, x):
        P = []
        for (i,p) in enumerate(self._params):
            if self._log[p]:
                P.append(N.exp(x[i]))
            else:
                P.append(x[i])
        return tuple(P)
    def plotPoints(self, plist, errlist):
        if self._plot2d:
            y = [p[1] for p in plist]
        else:
            y = errlist
        x = [p[0] for p in plist]
        self.__scatteroffsets.extend(zip(x,y))
        self.__collection.set_offsets(self.__scatteroffsets)
        # Directly applying color is faster than letting matplotlib handle colormaps (??)
        color = [self._colormap(1-e) for e in errlist]
        self.__facecolors.extend(color)
        self.__collection.set_facecolor(self.__facecolors)
        if len(self.__scatteroffsets)==len(plist):
            self.__axes.add_collection(self.__collection)
        self.__axes.update_datalim(self.__scatteroffsets)
        self.__axes.autoscale_view()
    def _evalP(self, p, plotImmediately=False):
        #p = self._xToP(x)
        if not p in self._errs:
            for (pp, param) in zip(p, self._params):
                setattr(self._clf, param, pp)
            self._errs[p] = self._psel(self._dset)
            if plotImmediately and self._plot:
                self.plotPoint([p], [self._errs[p]])
        return self._errs[p]
        
def _grid(argtup):
    """Returns a list of tuples containing every perumutation in 
    the tuple(args1, args2...) passed in

    ie returns [(arg1[0],arg2[0]), (arg1[0],arg2[1]),...]
    """
    tup = '(' 
    gen = ''
    for i in range(len(argtup)):
        tup += 'p%d,'%i
        gen += ' for p%d in argtup[%d]'%(i, i)
    tup += ')'
    return eval('['+tup+gen+']')
    
def makePsel(clf, psel):
    """Helper function modifies an existing classifier to automatically run
    parameter selection every time it's trained
    
    Perhaps should wrap into ProxyClassifier instead?
    WIP
    """
    # Perhaps this should be wrapped into a class of its own, passing getattr etc to self.__clf
    if hasattr(clf, '_psel'):
        Warning('Overwritting _psel attribute... hopefully this is intentional')
    
    clf._psel=psel # Allows changing psel instance
    clf.__selecting = False
    if not hasattr(clf, '_rawtrain'):#Prevents recursive wrapping
        clf._rawtrain = clf.train
        def train(self, dataset, *args, **kwargs):
            """Runs parameter selection, then trains the classifier as normal"""
            if not self.__selecting:
                self.__selecting=True # Prevents recursive calls!!
                self._psel(dataset)
                self.__selecting=False
            return self._rawtrain(dataset, *args, **kwargs)
        clf.train=clf._train.__class__(train, clf) # create as instancemethod
    def selection_summary(self):
        """Returns a string summarizing the last selection step"""
        s = '    --  parameter selection: %d%% to %d%% in %d seconds' %(100*(1.-self._psel.worst),
                                                                        100*(1-self._psel.best),
                                                                        self._psel.time)
        return s
    clf.selection_summary=clf._train.__class__(selection_summary, clf)
    
if __name__ == '__main__':
    # Example/test usage, since unittest is not written yet
    from mvpa.clfs.svm import RbfCSVMC, SVM
    from mvpa.clfs.sg.custom import CachedSVM, CachedRbfSVM
    from mvpa.misc.data_generators import normalFeatureDataset, dumbNormalFeatureBinaryDataset
    from mvpa.datasets.splitters import NFoldSplitter, HalfSplitter
    import pylab
    
    clf =  CachedRbfSVM()
    dset = dumbNormalFeatureBinaryDataset(perlabel=100, nchunks=2, snr=6)
    
    #dset = dumbFeatureBinaryDataset()
    cdset = clf.cache(dset)
    from mvpa.misc.plot import plotDecisionBoundary2D
    #pylab.ion()
    psel = ParameterSelection(clf, ('gamma', 'C'), HalfSplitter(), iterations=2, plot=True,manipulateClassifier=True)
    clf.train(cdset)
    def cacheme(newsamples):
        (cd, cs) = clf.cacheMultiple(dset, newsamples)
        return cs
    plotDecisionBoundary2D(dset, clf, dataCallback=cacheme)
    print clf.selection_summary()
    pylab.show()
    pass
