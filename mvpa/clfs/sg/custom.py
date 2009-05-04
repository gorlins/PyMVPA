#!/usr/bin/python
#coding:utf-8

"""Module implementing a custom kernel in Shogun

Useful ie for any precalculated kernel, for model selection, etc

WIP
"""

import numpy as N
from mvpa.datasets import Dataset
from mvpa.clfs.sg.svm import SVM, _tosg, _setdebug
from shogun.Kernel import CustomKernel
from shogun.Features import Labels
import operator

from mvpa.base import warning
if __debug__:
    from mvpa.base import debug
    
class CachedSVM(SVM):
    """A classifier which can cache the kernel matrix to enable fast training,
    retraining, cross-validation, etc
    
    Inits like a normal shogun SVM, so you can use any normal kernel or
    implementation.
    
    Note that the kernel cache will become invalid should a kernel parameter
    change (ie gamma), and the data must be recached for the new parameter to 
    have an effect.  C is not a kernel parameter, so it may be changed freely.
    
    Subclasses may deal with changing kernel parameters (eg, CachedRbfSVM)
    
    These classifiers must run on a special dataset that is created 
    with cached = self.cache(dset) or (cd1, cd2,...cdn) = self.cacheMultiple(...)
    
    Beware using default parameters, like C=-1, because when the Classifier
    calls _getDefault(Something) it will calculate the parameter on the cached data
    index, instead of the real data!!  _getDefaultC is overridden in this class 
    because it depends on the norm of the data.  _getDefaultGamma doesn't depend
    on the data (just the number of labels) so it doesn't need to be overridden 
    here, but if other defaults are added, they should be overriden in this class.
    """
    #def __init__(self, *args, **kwargs):
        #SVM.__init__(self, *args, **kwargs)
    def assertCachedData(self, samples):
        if isinstance(samples, Dataset):
            samples = samples.samples
            #self.assertCachedData(samples.samples)
        else:
            #try:
            assert samples.dtype==int
            try:
                assert samples.shape[1]==1
                assert all(N.logical_and(samples>=0, samples < self.__cached_kernel.shape[0]))
            except Exception:
                raise AssertionError('Cached classifier running on non-cached data')# Doesn't have cached kernel
            #except AssertionError:
                #raise AssertionError('Cached classifier running on non-cached data')
    def __cacheLHS(self, lhs):
        """Grabs lhs from the kernel matrix and caches it"""
        self.__cached_lhs = self.__cached_kernel.take(lhs.ravel(), axis=0)
        
    def __getCachedRHS(self, rhs):
        """Similar to __getChachedMatrix, but leaves the LHS intact (maybe
        faster for predicting, though untested)
        """
        return self.__cached_lhs.take(rhs.ravel(), axis=1)
    def __getCachedMatrix(self, lhs,rhs=None):
        """Creates a full kernel matrix from the cached data
        
        lhs, rhs must be (Nx1) samples representing cached data via indeces
        If not specified, rhs=lhs (square kernel matrix)
        """
        if isinstance(lhs, Dataset):
            lhs=lhs.samples
        if rhs is None:
            rhs = lhs
        if isinstance(rhs, Dataset):
            rhs=rhs.samples
        return self.__cached_kernel.take(lhs.ravel(), axis=0).take(rhs.ravel(), axis=1)
    def cacheMultiple(self, *dsets):
        """Caches multiple datasets simultaneously, returning a list with cached
        items corresponding to the inputs
        
        This can be used to predict new (uncached) data without retraining if the
        first dataset is that which it was trained on, as it will update the lhs
        with those indeces.  Retraining will overwrite the lhs, so don't worry
        about it if this is not your intended behavior
        """
        def asdata(d):
            if isinstance(d, Dataset):
                return d.samples, d.nsamples
            return d, d.shape[0]
        def ascache(c, d):
            if isinstance(d, Dataset):
                dout = d.selectFeatures([0])
                dout.setSamplesDType(int)
                dout.samples = c
            else:
                dout=c
            return dout
        alld = map(asdata, dsets)
        pured = [d[0] for d in alld]
        puren = N.asarray([d[1] for d in alld])
        allc = self.cache(N.concatenate(pured))
        dout = map(ascache, N.split(allc, puren.cumsum()[:-1]), dsets)
        self.__cacheLHS(dout[0].samples)
        return dout
        
    def cache(self, dset):
        """Generates a kernel for the dataset

        Importantly, also returns a dataset with the kernel index ids instead of
        normal features.  Use this dset (or subsections of it) for this
        classifier instead of the original dataset
        """
        if isinstance(dset, N.ndarray):
            samples=dset
            dout = N.arange(samples.shape[0]).reshape((samples.shape[0], 1))
        elif isinstance(dset, Dataset):
            samples=dset.samples
            dout = dset.selectFeatures([0])
            dout.setSamplesDType(int)
            dout.samples = N.arange(dout.nsamples).reshape((dout.nsamples, 1))
        else:
            raise RuntimeError('Unknown datatype of class %s'%dset.__class__)
            
        # Init kernel
        #kargs = []
        #for arg in self._KERNELS[self._kernel_type_literal][1]:
            #value = self.kernel_params[arg].value
            ## XXX Unify damn automagic gamma value
            #if arg == 'gamma' and value == 0.0:
                #value = self._getDefaultGamma(dset)
            #kargs += [value]
            
        #sgdata = _tosg(samples)
        self.__cached_kernel = self.calculateFullKernel(dset)#  self._kernel_type(sgdata, sgdata, *kargs).get_kernel_matrix()
        return dout

    def calculateFullKernel(self, lhs, rhs=None, kernel_type_literal=None):
        """Calculates the full kernel for any known type,  using self's params"""
        if kernel_type_literal is None:
            kernel_type_literal = self._kernel_type_literal
        if isinstance(lhs, Dataset):
            lhs = lhs.samples
        lhs = _tosg(lhs)
        if rhs is None:
            rhs = lhs
        else:            
            if isinstance(rhs, Dataset):
                rhs = rhs.samples
            rhs = _tosg(rhs)
        
        # Init kernel
        kargs = []
        for arg in self._KERNELS[kernel_type_literal][1]:
            value = self.kernel_params[arg].value
            # XXX Unify damn automagic gamma value
            if arg == 'gamma' and value == 0.0:
                value = self._getDefaultGamma(dset)
            kargs += [value]
        return self._KERNELS[kernel_type_literal][0](lhs, rhs, *kargs).get_kernel_matrix()
    def cacheNewLhsKernel(self, lhs, rhs, kernel_type_literal=None):
        """Caches the lhs vs rhs matrix, stores in __cached_lhs, returns indexes for dout only
        
        useful if lhs is training data (nonindexed) and rhs is new test data
        """
        if isinstance(rhs, Dataset):
            dout = rhs.selectFeatures([0])
            dout.setSamplesDType(int)
            dout.samples = N.arange(dout.nsamples).reshape((dout.nsamples, 1))
        if isinstance(rhs, N.ndarray):
            dout = N.arange(rhs.shape[0]).reshape((rhs.shape[0], 1))
        self.__cached_lhs = self.calculateFullKernel(lhs, rhs=rhs, kernel_type_literal=kernel_type_literal)
        return dout
        
    def __makeKernelFromFull(self, full, train=False):
        """Creates a CustomKernel set with self.__cached_kernel
        if train, also calls IdentityKernelNormalizer (dunno if this does anything,
        but it's in SVM parent in training)
        """
        from mvpa import externals
        kernel = CustomKernel()
        
        kernel.set_full_kernel_matrix_from_full(full)
        if train and externals.exists('sg >= 0.6.4'):
            from shogun.Kernel import IdentityKernelNormalizer
            kernel.set_normalizer(IdentityKernelNormalizer())
        self._SVM__condition_kernel(kernel)
        return kernel
    def _train(self, dataset):
        
        # Builds the kernel from cached
        self.assertCachedData(dataset.samples)
        self._SVM__kernel = self.__makeKernelFromFull(self.__getCachedMatrix(dataset.samples), train=True)
        self.__cacheLHS(dataset.samples)
        
        # Set appropriate values so that SVM._train does not recreate kernel
        self.retrainable = True
        if self._changedData is None:
            self._changedData = {}
        self._changedData['traindata'] = False
        self._changedData['kernel_params'] = False # Can't use these now
        self._SVM__svm = None # Forces super to create new svm
        
        SVM._train(self, dataset) # Super _train
        self.retrainable = False
    """#Copied from SVM, not needed if above works
        # Sets kernel with cached
        
        ul=None
        if 'regression' in self._clf_internals:
            labels_ = N.asarray(dataset.labels, dtype='double')
        else:
            ul = dataset.uniquelabels
            ul.sort()

            if len(ul) == 2:
                # assure that we have -1/+1
                _labels_dict = {ul[0]:-1.0, ul[1]:+1.0}
            elif len(ul) < 2:
                raise ValueError, "we do not have 1-class SVM brought into SG yet"
            else:
                # can't use plain enumerate since we need them swapped
                _labels_dict = dict([ (ul[i], i) for i in range(len(ul))])

            # reverse labels dict for back mapping in _predict
            _labels_dict_rev = dict([(x[1], x[0])
                                     for x in _labels_dict.items()])

            # bind to instance as well
            self._labels_dict = _labels_dict
            self._labels_dict_rev = _labels_dict_rev

            # Map labels
            #
            # TODO: top level classifier should take care about labels
            # mapping if that is needed
            labels_ = N.asarray([ _labels_dict[x] for x in dataset.labels ], dtype='double')

        labels = Labels(labels_)
        _setdebug(labels, 'Labels')


        Cs = None    
        # SVM
        if self.params.isKnown('C'):
            C = self.params.C
            if not operator.isSequenceType(C):
                # we were not given a tuple for balancing between classes
                C = [C]
                
                Cs = list(C[:])               # copy
                for i in xrange(len(Cs)):
                    if Cs[i]<0:
                        Cs[i] = self._getDefaultC(dataset.samples)*abs(Cs[i])
                        
        # Choose appropriate implementation
        svm_impl_class = self._SVM__get_implementation(ul)

        if self._svm_impl in ['libsvr', 'svrlight']:
            # for regressions constructor a bit different
            self._SVM__svm = svm_impl_class(Cs[0], self.params.epsilon, self._SVM__kernel, labels)
        elif self._svm_impl in ['krr']:
            self._SVM__svm = svm_impl_class(self.params.tau, self._SVM__kernel, labels)
        else:
            self._SVM__svm = svm_impl_class(Cs[0], self._SVM__kernel, labels)
            self._SVM__svm.set_epsilon(self.params.epsilon)
        if Cs is not None and len(Cs) == 2:
            if __debug__:
                debug("SG_", "Since multiple Cs are provided: %s, assign them" % Cs)
            self._SVM__svm.set_C(Cs[0], Cs[1])

        self.params.reset()  # mark them as not-changed
        newsvm = True
        _setdebug(self._SVM__svm, 'SVM')
        # Set optimization parameters
        if self.params.isKnown('tube_epsilon') and \
               hasattr(self._SVM__svm, 'set_tube_epsilon'):
            self._SVM__svm.set_tube_epsilon(self.params.tube_epsilon)
        self._SVM__svm.parallel.set_num_threads(self.params.num_threads)
        
        self._SVM__svm.train()
        self._trained_dataset = dataset.copy()

        # Report on training
        if self.states.isEnabled('training_confusion'):
            trained_labels = self._SVM__svm.classify().get_labels()
        else:
            trained_labels = None

        if self.regression and self.states.isEnabled('training_confusion'):
            self.states.training_confusion = self._summaryClass(
                targets=dataset.labels,
                predictions=trained_labels)
    """
    def _predict(self, samples):
    
        # Builds the kernel from cached
        self.assertCachedData(samples)
        self._SVM__kernel_test = self.__makeKernelFromFull(self.__getCachedRHS(samples))
        
        # Set appropriate values so that SVM._train does not recreate kernel
        self.retrainable = True
        if self._changedData is None:
            self._changedData = {}
        self._changedData['traindata'] = False
        self._changedData['kernel_params'] = False # Can't use these now
        self._changedData['testdata'] = False
        
        predictions = SVM._predict(self, samples)
        self.retrainable = False
        return predictions
        
    """ # Copied/modified from SVM._predict, not needed if above works
    self.assertCachedData(samples)
    self._SVM__kernel.set_full_kernel_matrix_from_full(self.__getCachedRHS(samples))
    self._SVM__condition_kernel(self._SVM__kernel)
    self._SVM__svm.set_kernel(self._SVM__kernel)
    values_ = self._SVM__svm.classify()
    if values_ is None:
        raise RuntimeError, "We got empty list of values from %s" % self

    values = values_.get_labels()
    if ('regression' in self._clf_internals):
        predictions = values
    else:
        # local bindings
        _labels_dict = self._labels_dict
        _labels_dict_rev = self._labels_dict_rev

        if len(_labels_dict) == 2:
            predictions = 1.0 - 2*N.signbit(values)
        else:
            predictions = values

        # assure that we have the same type
        label_type = type(_labels_dict.values()[0])

        # remap labels back adjusting their type
        predictions = [_labels_dict_rev[label_type(x)]
                       for x in predictions]

    

    # store state variable
    self.values = values

    return predictions
    """
    
    

    def _getDefaultC(self, data):
        #Overriden to not depend on data, since it may not be cached
        try:
                self.assertCachedData(data)
                d = N.sqrt(self.__getCachedMatrix(data).diagonal())
                d = d.reshape((d.size, 1))
                return SVM._getDefaultC(self, d)
        except AssertionError:
            warning("Asking for default C on non-cached data.  Assigning 1.0")
            return 1.0



class CachedRbfSVM(CachedSVM):
    """A cached SVM with an Rbf kernel
    
    Automagically updates the kernel when gamma is changed, avoiding the need
    to recalculate it
    
    One difference between this and normal SVM is gamma cannot be 0 (to
    automagically set default gamma) due to api complications. 
    You may still call getDefaultGamma instead.

    The actual kernel used is a cached linear kernel, so don't expect it to
    have an identical interface to a normal RbfSVM.  This is done so that 
    changing gamma only involves converting the linear kernel to a distance,
    dividing by gamma, and doing the exponent - no new dot product is needed.
    
    Due to the implementation, the lhs is not updated automatically when gamma
    changes.  The lhs is extracted from the full matrix during training, so
    it is okay that it is not explicitly updated with the full matrix, since 
    you should retrain if you change gamma anyway.  Perhaps there's a way to do
    that caching as a view instead of copy, but I haven't done that yet.
    
    NB: The kernel matrix is stored twice (once for the linear distance, once
    for the Rbf), so beware with huge sample sizes.
    """
    _KNOWN_KERNEL_PARAMS = ['gamma']
    def __init__(self, kernel_type='rbf', **kwargs):
        if not kernel_type == 'rbf':
            raise RuntimeError('CachedRbfSVM must have kernel type ''rbf'', not ''%s'''%kernel_type)
        CachedSVM.__init__(self, kernel_type='linear', **kwargs)
        #self.gamma=1.
    def cache(self, d):
        dout = CachedSVM.cache(self, d)
        self.__explinear = -self.__linearToDist(self._CachedSVM__cached_kernel)
        self.__updateCache()
        return dout
    
    def cacheNewLhsKernel(self, lhs, rhs=None, kernel_type_literal=None):
        # Overrides because we need the rbf here, since we can't calculate the distance
        #between points from linear product without calculating whole kernel matrix
        return CachedSVM.cacheNewLhsKernel(self, lhs, rhs, kernel_type_literal='rbf')
        
    def __updateCache(self):
        """Creates the Rbf kernel from the linear distance"""
        try:
            self._CachedSVM__cached_kernel = N.exp(self.__explinear/self.gamma)#self.__distToRbf(self.__linear, self.gamma)
        except AttributeError: # in case kernel not calculated yet
            pass
        
    @staticmethod
    def __linearToDist(k):
        """Convert a dot product kernel matrix to squared Euclidean distance"""
        return N.diag(k).reshape((1, k.shape[1])) - 2*k + N.diag(k).reshape((k.shape[0], 1))
    
    @staticmethod
    def __distToRbf(k, g):
        return N.exp(-k/g)
        
    def __setattr__(self, attr, value):
        """Automagically adjust kernel for new value of gamma, otherwise set normally"""
        if attr=='gamma' and value != self.gamma:
            CachedSVM.__setattr__(self, attr, value)
            self.__updateCache()
        else:
            CachedSVM.__setattr__(self, attr, value)
    def _getDefaultC(self, dataset):
        #algorithm always returns 1.0 since diagonal is always 1.0 in Rbf
        # this also prevents some errors
        return 1.0
    def _getDefaultGamma(self, dataset):
        
        # Have to override default which checks for known parameters.  maybe unify api at some point in future
        value = 1.0 / len(dataset.uniquelabels)
        if __debug__:
            debug("SVM", "Default Gamma is computed to be %f" % value) # have to find this module and import it
            pass
        return value


#from shogun.Features import RealFeatures, Labels
#from shogun.Kernel import CustomKernel, LinearKernel
#from shogun.Classifier import LibSVM

#from mvpa.clfs.sg.svm import _tosg, SVM
if __name__ == '__main__':
    from mvpa.misc.data_generators import normalFeatureDataset
    from mvpa.datasets.splitters import NFoldSplitter
    from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
    from mvpa.clfs.transerror import TransferError
    
    import time
    s = NFoldSplitter()
    nfeatures =10000
    d = normalFeatureDataset(perlabel=100, nfeatures=nfeatures, nchunks=10, 
                             means=[N.random.randn(nfeatures), N.random.randn(nfeatures)], snr=10./nfeatures)
    
    
    cvNormal = CrossValidatedTransferError(TransferError(SVM(C=0.1)), s)
    
    csvm = CachedSVM(C=0.1)    
    csvm.gamma = .001
    cvCached = CrossValidatedTransferError(TransferError(csvm), s)
    
    t1 = time.time()
    cd = csvm.cache(d)
    (cd1, cd2) = s(cd).next()
    t2 = time.time()
    cachederr=cvCached(cd)
    t3 = time.time()
    
    t4 = time.time()
    normerr = cvNormal(d)
    t5 = time.time()
    
    print 'Kernel cache: %fs'%(t2-t1)
    print 'Cached CV: %fs, err %i%%'%(t3-t2, cachederr*100)
    print 'Normal CV: %fs, err %i%%'%(t5-t4, normerr*100)
    
    import mvpa.tests.test_customkernels
    mvpa.tests.test_customkernels.run()
    ## Errors with psel currently... runs but not selecting minimum??
    #from mvpa.clfs.parameter_selector import ParameterSelection
    #crbf = CachedRbfSVM()
    #ParameterSelection(crbf, ('C', 'gamma'), s, manipulateClassifier=True)
    #cd = crbf.cache(d)
    #crbf.train(cd)
    #from pylab import show
    #show()
    #print crbf._psel.best
    #C=1
    #dim=7
    #from mvpa.misc.data_generators import dumbFeatureBinaryDataset
    #d=dumbFeatureBinaryDataset()
    #lab=N.sign(2*d.labels.astype(float)-1)
    #data = d.samples.astype(float)
    #symdata=N.dot(data, data.T)
    
    #from shogun.Kernel import LinearKernel
    #from shogun.Classifier import LibSVM
    #C=0.1
    #f = _tosg(d.samples)
    #lk = LinearKernel(f, f)
    #kernel=CustomKernel()
    #kernel.set_full_kernel_matrix_from_full(lk.get_kernel_matrix())
    #labels=Labels((2*d.labels-1).astype(float))
    #lsvm=LibSVM(C, lk, labels)
    #lsvm.train()
    #svm=LibSVM(C, kernel, labels)
    #svm.train()
    #out=svm.classify().get_labels()
    #lout=lsvm.classify().get_labels()
    pass
