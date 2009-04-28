#!/usr/bin/python
#coding:utf-8

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
    def __init__(self, classifier, params, splitter, scales = None, defaults=None, 
                 cv=None, te=None,
                 log=True, factors = 2., iterations=2, plot=True, nrows=1, ncols=1, 
                 manipulateClassifier=False):
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
                self._scales[param]=N.arange(-16., 17., 2., dtype='double')
                
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
        """Runs the parameter selection on a given mvpa dataset, internally setting 
        classifier parameters
        
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
                    except Error:
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
                
        errs = {}
        self.worst = 0.
        
        # Creates figure
        if self._plot:
            import pylab
            shouldTurnOff = not pylab.isinteractive()
            pylab.ion()
            px = params[0]
            if self._plot2d:
                py = params[1]
            #shouldTurnOff = not pylab.isinteractive()
            #pylab.ion()
            ##pylab.hold(False)
            
            pylab.subplot(self._nrows, self._ncols, self._n)
            #pylab.ioff()
            if self._plot2d:
                if self._log[px] and not self._log[py]:
                    pylab.semilogx()
                elif self._log[py] and not self._log[px]:
                    pylab.semilogy()
                elif self._log[px] and self._log[py]:
                    pylab.loglog()
                else:
                    pass#pylab.Figure()
                pylab.ylabel(py)
            else:
                if self._log[px]:
                    pylab.semilogx()
                else:
                    pass
                pylab.ylabel('CV Error')
            
            pylab.xlabel(px)
            pylab.title(title)
            pylab.hold(True)
            init=True
            sc = pylab.scatter([.5], [.5], [0.], 
                               c=[1.], cmap=pylab.cm.Paired, vmin=0, vmax=1, edgecolor='None', marker='s')
            xarray = []
            yarray = []
            counter=0
            pylab.draw()
     

        # Runs the optimization
        err = N.zeros(tuple([len(s) for s in scales.itervalues()]))
        for i in range(self._iter):
            # Build grids
            grids = {}
            for p in params:
                if self._log[p]:
                    grids[p] = origins[p]*2.**(scales[p]+offsets[p])
                else:
                    grids[p] = scales[p]
                    
            pgrid = _grid(tuple([g for g in grids.values()]))
            for (i, pvals) in enumerate(pgrid):
                for (ii,p) in enumerate(params):
                    setattr(self._clf, p, pvals[ii])
                    
                # Grabs or calculates the error at this pvals
                if errs.has_key(pvals):
                    err.ravel()[i] = errs[pvals]
                else:
                    # Calculates error e
                    e = self._psel(dset)
                    # Uses product of training and testing errors to ensure good parameters!
                    #e = 1-(1-0*self._psel.training_confusion.error)*(1-self._psel.confusion.error)
                    
                    # Stores error e
                    err.ravel()[i]=e
                    errs[pvals] = err.ravel()[i]
                    
                    # Updates plot
                    if self._plot:
                        xarray.append(pvals[0])
                        if self._plot2d:
                            yarray.append(pvals[1])
                            sc.set_offsets(N.concatenate((sc._offsets,[pvals])))
                        else:
                            yarray.append(errs[pvals])
                            sc.set_offsets(N.concatenate((sc._offsets, [[pvals[0], errs[pvals]]])))
                        
                        sc._sizes = N.concatenate((sc._sizes,N.maximum([100.*errs[pvals]**2], 10)))
                        sc.set_array(N.concatenate((sc.get_array(),[errs[pvals]]), axis=0))
                    
                        # Updates axes
                        minx = min(xarray)
                        maxx = max(xarray)
                        miny = min(yarray)
                        maxy = max(yarray)
                        
                        if self._log[px]:
                            minx*=.5
                            maxx*=2.
                        else:
                            pad = (maxx-minx)/20.
                            minx-=pad
                            maxx+=pad
                        if self._plot2d and self._log[py]:
                            miny*=.5
                            maxy*=2.
                        else:
                            pad = (maxy-miny)/20.
                            
                            miny-=pad
                            maxy+=pad
                        if not self._plot2d:
                            miny = 0.
                            maxy = 1.
                        corners = (minx, miny), (maxx, maxy)
                        sc.axes.update_datalim(corners)
                        sc.axes.autoscale_view()
                                                    
                # Limits scatter drawing to once per column, for speed
                if self._plot:
                    counter+=1
                    if self._plot2d:
                        n = len(grids[py])
                    else:
                        n = len(grids[px])
                    if counter%n == 0:
                        #pylab.draw()
                        if init:
                            init=False
                            #pylab.colorbar()
            
            mins = err==err.min()
            if sum(mins.ravel()) == 1:
                eid = err.argmin()
            else:
                #g = scipy.ndimage.morphological_gradient(err, size=[3]*err.ndim)
                sigma=1
                while sum(mins.ravel()) > 1 and sigma<5:
                    g = gaussian_filter(err, sigma,cval=1., mode='constant')
                    mins = N.where(mins, g, g.max())==g.min()
                    sigma+=1
                eid = N.where(mins, g, g.max()).argmin()
            eidunravel = N.unravel_index(eid, err.shape)
            self.best = err.ravel()[eid]
            
            # Set best parameters, update offsets, rescales
            for (i, p) in enumerate(params):
                setattr(self._clf, p, pgrid[eid][i])
                if self._log[p]:
                    # Recenter scales on best point
                    offsets[p]+=scales[p][eidunravel[i]]
                    
                    # Correct for non-centered scales
                    offsets[p]-=scales[p][N.floor(len(scales[p])/2.)]/self._factors[p]
                    
                    # Rescale
                    scales[p]/=self._factors[p] 
               
            self.worst = max(self.worst, err.max())
        self.time=time()-t0
        if self._plot:
            pylab.draw()
            if self._plot2d:
                pylab.axhline(getattr(self._clf, py), color='k')
            else:
                pylab.axhline(errs[(getattr(self._clf,px),)], color='k')
            pylab.axvline(getattr(self._clf, px), color='k')
            pylab.draw()
            if shouldTurnOff:
                pylab.ioff()
            
            
def _grid(argtup):
    """Returns a list of tuples containing every perumutation in 
    the tuple(args1, args2...) passed in

    ie returns [(arg1[0],arg2[0]), (arg1[0],arg2[1]),...]
    """
    tup = '(' 
    gen = ''
    for i in range(len(argtup)):
        tup+='p%d,'%i
        gen += ' for p%d in argtup[%d]'%(i, i)
    tup += ')'
    return eval('['+tup+gen+']')
    
def makePsel(clf, psel):
    """Helper function modifies an existing classifier to automatically run
    parameter selection every time it's trained
    
    WIP
    """
    if not hasattr(clf, '_rawtrain'):#Prevents recursive wrapping
        clf._psel=psel
        clf._rawtrain = clf.train
        clf.__selecting = False
        def train(self, dataset, *args, **kwargs):
            """Runs parameter selection, then trains the classifier as normal"""
            if not self.__selecting:
                self.__selecting=True # Prevents recursive calls!!
                self._psel(dataset)
                self.__selecting=False
            return self._rawtrain(dataset, *args, **kwargs)
        clf.train=clf._train.__class__(train, clf)
        def selection_summary(self):
            """Returns a string summarizing the last selection step"""
            s = '    --  parameter selection: %d%% to %d%% in %d seconds' %(100*(1.-self._psel.worst),
                                                                            100*(1-self._psel.best),
                                                                            self._psel.time)
            return s
        clf.selection_summary=clf._train.__class__(selection_summary, clf)
    
if __name__ == '__main__':
    from mvpa.clfs.svm import RbfCSVMC
    from mvpa.misc.data_generators import normalFeatureDataset
    from mvpa.datasets.splitters import NFoldSplitter
    import pylab
    
    clf = RbfCSVMC()
    dset = normalFeatureDataset(nfeatures=2, means=[[0,1], [1,0]])
    
    psel = ParameterSelection(clf, ('C', 'gamma'), NFoldSplitter(), manipulateClassifier=True)
    clf.train(dset)
    print clf.selection_summary()
    pylab.show()
    pass
