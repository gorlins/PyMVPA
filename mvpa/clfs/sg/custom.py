#!/usr/bin/python
#coding:utf-8

"""Module implementing a custom kernel in Shogun

Useful ie for any precalculated kernel, for model selection, etc

WIP
"""

import numpy as N
from mvpa.clfs.sg.svm import SVM, _tosg, _setdebug
from shogun.Kernel import CustomKernel
from shogun.Features import Labels
import operator

class CachedSVM(SVM):
    """A classifier which can cache the kernel matrix to enable fast
    cross-validation
    
    Inits like a normal shogun SVM, so you can use any normal kernel or
    implementation.
    
    However, it takes a special dataset that is created with self.cache(dset)
    
    """
    def __init__(self, *args, **kwargs):
        SVM.__init__(self, *args, **kwargs)
        self.__kernel = CustomKernel()
    def assertCachedData(self, samples):
        try:
            assert True
        except AssertionError:
            raise RuntimeError('Cached classifier running on non-cached data')
    def __getCachedMatrix(self, samples, s2=None):
        self.assertCachedData(samples)
        if s2 is None:
            s2 = samples
        else:
            self.assertCachedData(s2)
        return self.__cached_kernel.take(samples.ravel(), axis=0).take(s2.ravel(), axis=1)
    def cache(self, dset):
        """Generates a kernel for the dataset

        Importantly, also returns a dataset with the kernel index ids instead of
        normal features.  Use this dset (or subsections of it) for this
        classifier instead of the original dataset
        """
        if isinstance(dset, N.ndarray):
            samples=dset
            dout = N.arange(samples.shape[0]).reshape((samples.shape[0], 1))
        else:
            samples=dset.samples
            dout = dset.selectFeatures([0])
            dout.setSamplesDType(int)
            dout.samples = N.arange(dout.nsamples).reshape((dout.nsamples, 1))
            
        # Init kernel
        kargs = []
        for arg in self._KERNELS[self._kernel_type_literal][1]:
            value = self.kernel_params[arg].value
            # XXX Unify damn automagic gamma value
            if arg == 'gamma' and value == 0.0:
                value = self._getDefaultGamma(dataset)
            kargs += [value]
            
        self.__traindata = _tosg(dset.samples)
        self.__cached_kernel = self._kernel_type(self.__traindata, self.__traindata,
                                                 *kargs).get_kernel_matrix()
        return dout
    def _train(self, dataset):
        # Sets kernel with cached
        self.__kernel.set_full_kernel_matrix_from_full(self.__getCachedMatrix(dataset.samples))
        
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
        self.__kernel.set_full_kernel_matrix_from_full(self.__getCachedMatrix(self._trained_dataset.samples, samples))
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





#from shogun.Features import RealFeatures, Labels
#from shogun.Kernel import CustomKernel, LinearKernel
#from shogun.Classifier import LibSVM

#from mvpa.clfs.sg.svm import _tosg, SVM
if __name__ == '__main__':
    from mvpa.misc.data_generators import dumbFeatureBinaryDataset, normalFeatureDataset
    from mvpa.datasets.splitters import NFoldSplitter
    from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
    from mvpa.clfs.transerror import TransferError
    
    import time
    s = NFoldSplitter()
    d = normalFeatureDataset(perlabel=500, nfeatures=2000, nchunks=4)
    
    cvNormal = CrossValidatedTransferError(TransferError(SVM()), s)
    
    csvm = CachedSVM()    
    cvCached = CrossValidatedTransferError(TransferError(csvm), s)
    
    t1 = time.time()
    cd = csvm.cache(d)
    (cd1, cd2) = s(cd).next()
    t2 = time.time()
    cvCached(cd)
    t3 = time.time()
    
    t4 = time.time()
    cvNormal(d)
    t5 = time.time()
    
    print 'Kernel cache: %f s'%(t2-t1)
    print 'Cached CV: %f s'%(t3-t2)
    print 'Normal CV: %f s'%(t5-t4)

    #C=1
    #dim=7
    #from mvpa.misc.data_generators import dumbFeatureBinaryDataset
    #d=dumbFeatureBinaryDataset()
    #lab=N.sign(2*d.labels.astype(float)-1)
    #data = d.samples.astype(float)
    #symdata=N.dot(data, data.T)
    
    #f = _tosg(data)
    #lk = LinearKernel(f, f)
    #kernel=CustomKernel()
    #kernel.set_full_kernel_matrix_from_full(symdata)
    #labels=Labels(lab)
    #lsvm=LibSVM(C, lk, labels)
    #lsvm.train()
    #svm=LibSVM(C, kernel, labels)
    #svm.train()
    #out=svm.classify().get_labels()
    #lout=lsvm.classify().get_labels()
    pass
