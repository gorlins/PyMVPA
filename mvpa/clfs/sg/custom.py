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

class CachedSVM(SVM):
    """A classifier which can cache the kernel matrix to enable fast training,
    retraining, cross-validation, etc
    
    Inits like a normal shogun SVM, so you can use any normal kernel or
    implementation.
    
    However, it takes a special dataset that is created with self.cache(dset)
    
    Beware using default (kernel) parameters, like C=0, because when the Classifier
    calls _getDefault(Something) it may send the cached index instead of the 
    real data!!
    """
    def __init__(self, *args, **kwargs):
        SVM.__init__(self, *args, **kwargs)
        self.__kernel = CustomKernel()
    def assertCachedData(self, samples):
        try:
            assert True
        except AssertionError:
            raise RuntimeError('Cached classifier running on non-cached data')
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
        if rhs is None:
            rhs = lhs
        return self.__cached_kernel.take(lhs.ravel(), axis=0).take(rhs.ravel(), axis=1)
    def cacheMultiple(self, *dsets):
        """Caches multiple datasets simultaneously, returning a list with cached
        items corresponding to the inputs"""
        def asdata(d):
            if isinstance(d, Dataset):
                return d.samples, d.nfeatures
            return d, d.shape[0]
        def ascache(c, d):
            if isinstance(d, Dataset):
                dout = d.selectFeatures([0])
                dout.setSamplesDType(int)
                dout.samples = N.arange(dout.nsamples).reshape((dout.nsamples, 1))
            else:
                dout=N.arange(d.shape[0]).reshape((d.shape[0], 1))
            return dout
        alld = map(asdata, dsets)
        pured = [d[0] for d in alld]
        puren = N.asarray([d[1] for d in alld])
        allc = self.cache(N.concatenate(pured))
        return map(ascache, N.split(allc, puren.cumsum()[:-1]), dsets)
        
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
        kargs = []
        for arg in self._KERNELS[self._kernel_type_literal][1]:
            value = self.kernel_params[arg].value
            # XXX Unify damn automagic gamma value
            if arg == 'gamma' and value == 0.0:
                value = self._getDefaultGamma(dset)
            kargs += [value]
            
        sgdata = _tosg(samples)
        self.__cached_kernel = self._kernel_type(sgdata, sgdata, *kargs).get_kernel_matrix()
        return dout
    def _train(self, dataset):
        self.assertCachedData(dataset.samples)
        # Sets kernel with cached
        self.__kernel.set_full_kernel_matrix_from_full(self.__getCachedMatrix(dataset.samples))
        self.__cacheLHS(dataset.samples)
        
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
            self.__svm = svm_impl_class(Cs[0], self.params.epsilon, self.__kernel, labels)
        elif self._svm_impl in ['krr']:
            self.__svm = svm_impl_class(self.params.tau, self.__kernel, labels)
        else:
            self.__svm = svm_impl_class(Cs[0], self.__kernel, labels)
            self.__svm.set_epsilon(self.params.epsilon)
        if Cs is not None and len(Cs) == 2:
            if __debug__:
                debug("SG_", "Since multiple Cs are provided: %s, assign them" % Cs)
            self.__svm.set_C(Cs[0], Cs[1])

        self.params.reset()  # mark them as not-changed
        newsvm = True
        _setdebug(self.__svm, 'SVM')
        # Set optimization parameters
        if self.params.isKnown('tube_epsilon') and \
               hasattr(self.__svm, 'set_tube_epsilon'):
            self.__svm.set_tube_epsilon(self.params.tube_epsilon)
        self.__svm.parallel.set_num_threads(self.params.num_threads)
        
        self.__svm.train()
        self._trained_dataset = dataset.copy()

        # Report on training
        if self.states.isEnabled('training_confusion'):
            trained_labels = self.__svm.classify().get_labels()
        else:
            trained_labels = None

        if self.regression and self.states.isEnabled('training_confusion'):
            self.states.training_confusion = self._summaryClass(
                targets=dataset.labels,
                predictions=trained_labels)
    
    def _predict(self, samples):
        self.assertCachedData(samples)
        self.__kernel.set_full_kernel_matrix_from_full(self.__getCachedRHS(samples))
        self._SVM__condition_kernel(self.__kernel)
        self.__svm.set_kernel(self.__kernel)
        values_ = self.__svm.classify()
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
    def __init__(self, kernel_type='rbf', **kwargs):
        if not kernel_type == 'rbf':
            raise RuntimeError('CachedRbfSVM must have kernel type ''rbf'', not ''%s'''%kernel_type)
        CachedSVM.__init__(self, kernel_type='linear', **kwargs)
        self.gamma=1.
    def cache(self, d):
        dout = CachedSVM.cache(self, d)
        self.__linear = self.__linearToDist(self._CachedSVM__cached_kernel)
        self.__updateCache()
        return dout
    def __updateCache(self):
        """Creates the Rbf kernel from the linear distance"""
        try:
            self._CachedSVM__cached_kernel = self.__distToRbf(self.__linear, self.gamma)
        except AttributeError: # in case kernel not calculated yet
            pass
        
    @staticmethod
    def __linearToDist(k):
        """Convert a dot product kernel matrix to squared Euclidean distance"""
        return N.diag(k).reshape((1, k.shape[1])) - 2*k+ N.diag(k).reshape((k.shape[0], 1))
    
    @staticmethod
    def __distToRbf(k, g):
        return N.exp(-k/g)
        
    def __setattr__(self, attr, value):
        """Automagically adjust kernel for new value of gamma, otherwise set normally"""
        CachedSVM.__setattr__(self, attr, value)
        if attr=='gamma':
            self.__updateCache()
            
    def _getDefaultGamma(self, dataset):
        
        # Have to override default which checks for known parameters.  maybe unify api at some point in future
        value = 1.0 / len(dataset.uniquelabels)
        if __debug__:
            #debug("SVM", "Default Gamma is computed to be %f" % value) # have to find this module and import it
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
