# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

"""Module implementing a custom kernel in Shogun

Useful ie for any precalculated kernel, for model selection, etc

WIP
"""

import numpy as N
from mvpa.datasets import Dataset
from mvpa.clfs.sg.svm import SVM, _tosg
from shogun.Kernel import CustomKernel, IdentityKernelNormalizer
#from shogun.Features import Labels
#import operator

from mvpa.base import warning
if __debug__:
    from mvpa.base import debug
    
class CachedSVM(SVM):
    """A classifier which can cache the kernel matrix to enable fast training,
    retraining, cross-validation, MC null distribution estimates, etc.
    
    The speed benefits of using this class vs a traditional SVM will only be 
    realized if you train or predict the same data more than once.  The more 
    you retrain (ie N-fold cross validation, more MC samples) and the slower
    your kernel is to calculate (many features) the more this class is for you.
    I.e., this class transforms a normal problem from O(t*P) to O(t) + O(P) for 
    t trials and P features
    
    Inits like a normal shogun SVM, so you can use any normal kernel or
    implementation.
    
    Note that the kernel cache will become invalid should a kernel parameter
    change (ie gamma), and the data must be recached for the new parameter to 
    have an effect.  C is not a kernel parameter, so it may be changed freely.
    
    Subclasses may deal with changing kernel parameters (eg, CachedRbfSVM) as
    required for fast parameter selection, etc.
    
    These classifiers must run on a special dataset that is created 
    with cached = self.cache(dset) or (cd1, cd2,...cdn) = self.cacheMultiple(..)
    
    Beware using default parameters, like C=-1, because when the Classifier
    calls _getDefault{Something} it will calculate the parameter on the cached 
    index, instead of the real data!!  _getDefaultC is overridden in this class 
    because it depends on the norm of the data.  _getDefaultGamma doesn't depend
    on the data (just the number of labels) so it doesn't need to be overridden 
    here, but if other defaults are added, they should be overriden.
    
    In terms of memory, this class stores the following matrices:
    
      self.__cached_kernel: (N x N) matrix storing ALL the currently cached
        kernel calculations, including training and testing data
      self.__cached_lhs: (n x N) matrix containing the last trained n samples
        against all N cached samples.  This is a COPY of the data.
        In the case that self.cacheNewLhsKernel is called, this matrix becomes
        (n x nr) where nr is the number of newly cached samples from the rhs.
        
      (This is in addition to the kernel matrices used internally by Shogun)
        
    Because of these considerations, this class should be avoided for extremely
    large N
    
    This setup means all data can be cached once up front, and any number of 
    retraining, CV, label-swapping, etc can reoccur without needing to
    recalculate the kernel (unless a kernel parameter is changed)
    
    DO NOT use this class if your FEATURES will change, ie for Recursive Feature
    Selection or a searchlight analysis, since you will need to recache the 
    kernel with each new featureset anyway (unless you need CV/more than one 
    training per selected feature set, this class will yield no benefit)
        
    XXX: it may be possible to do away with __cached_lhs and simply extract the
    lhs from the full matrix prior to each prediction, but this will prevent
    things like cacheNewLhsKernel
    """

    def assertCachedData(self, samples):
        """Raises an AssertionError if samples are not properly cached indeces
        
        :Parameters:
          samples: cached Dataset or numpy array of cached indeces
          
        :Returns:
          nothing
        """
        (samples, n) = self._asdata(samples)
        
        try:
            # Proper data type
            assert isinstance(samples, N.ndarray)
            assert samples.dtype == int
            assert len(samples.shape)==2
            assert samples.shape[1] == 1

            # Valid indeces
            try:
                valid = N.logical_and(samples >= 0,
                                      samples < self.__cached_kernel.shape[0])
            except AttributeError:
                raise AssertionError()
            assert valid.all()
            
        except AssertionError:
            raise AssertionError('Using non-cached data in a CachedSVM!!')
            
    def __cacheLHS(self, lhs):
        """Grabs lhs from the kernel matrix and caches it"""
        self.__cached_lhs = self.__cached_kernel.take(lhs.ravel(), axis=0)
        
    def __getCachedRHS(self, rhs):
        """Similar to __getChachedMatrix, but leaves the LHS intact (maybe
        faster for predicting, though untested)
        """
        return self.__cached_lhs.take(rhs.ravel(), axis=1)
    
    def __getCachedMatrix(self, lhs, rhs=None):
        """Creates a full kernel matrix from the cached data
        
        lhs, rhs must be (Nx1) samples representing cached data via indeces
        
        If not specified, rhs=lhs (square kernel matrix)
        """
        lhs, ln = self._asdata(lhs)
        if rhs is None:
            rhs = lhs
        rhs, rn = self._asdata(rhs)
            
        return self.__cached_kernel.take(lhs.ravel(), axis=0).take(rhs.ravel(),
                                                                   axis=1)

    @staticmethod
    def _asdata(d):
        """Takes dataset or ndarray and strips to raw data as necessary
        
        :Parameters:
          d: dataset or (N x M) numpy array
          
        :Returns:
          (samples, nsamples) where samples is d.samples or the raw numpy array
        """
        if isinstance(d, Dataset):
            return d.samples, d.nsamples
        return d, d.shape[0]
    
    @staticmethod
    def _ascache(c,d):
        """Takes a dataset or array and formats it as cached data
        
        :Parameters:
          c: 1D array of cache indeces
          d: either a full dataset or an ND-array
        
        :Returns:
          dout is either the cached indeces 'c' in the full dataset 'd', 
          retaining all labels, chunks, etc, or 'c' is returned directly as a
          (n x 1) numpy array of type int
        """
        if isinstance(d, Dataset):
            dout = d.selectFeatures([0]) # Drops to shape Nx1 
            dout.setSamplesDType(int)
            dout.samples[:,0] = c.ravel()[:]
        else:
            dout=c.astype(int).reshape(c.size, 1)
        return dout
    
    def cacheMultiple(self, *dsets):
        """Caches multiple datasets simultaneously, returning a list with cached
        items corresponding to the inputs
        
        This can be used to predict new (uncached) data without retraining if
        the first dataset is that which it was trained on, as it will update the
        lhs with those indeces.  Retraining will overwrite the lhs, so don't
        worry about it if this is not your intended behavior
        
        :Parameters:
          *dsets: any combination of Datasets or ND-arrays that can be cached
          via self.cache()
        
        :Returns:
          list matching input 1 to 1 of the cached indeces, either as Datasets
          or ND-arrays as per the input
        """
        
        alld = map(self._asdata, dsets)
        pured = [d[0] for d in alld]
        puren = N.asarray([d[1] for d in alld])
        allc = self.cache(N.concatenate(pured))
        dout = map(self._ascache, N.split(allc, puren.cumsum()[:-1]), dsets)
        self.__cacheLHS(dout[0].samples) # Treats first input as LHS
        return dout
        
    def cache(self, dset):
        """Generates a kernel from the dataset

        Importantly, also returns a dataset with the kernel index ids instead of
        normal features.  Use this dset (or subsections of it) for this
        classifier instead of the original dataset
        """
        (samples, n) = self._asdata(dset)            
        self.__cached_kernel = self.calculateFullKernel(samples)
        return self._ascache(N.arange(n), dset)

    def calculateFullKernel(self, lhs, rhs=None, kernel_type_literal=None):
        """Calculates the full kernel for any known type, using self's params"""
        if kernel_type_literal is None:
            kernel_type_literal = self._kernel_type_literal
        lhs, ln = self._asdata(lhs)
        sglhs = _tosg(lhs)
        if rhs is None:
            rhs = lhs
        else:            
            (rhs, rn) = self._asdata(rhs)
        sgrhs = _tosg(rhs)
        
        # Init kernel
        kargs = []
        for arg in self._KERNELS[kernel_type_literal][1]:
            value = self.kernel_params[arg].value
            # XXX Unify damn automagic gamma value
            if arg == 'gamma' and value == 0.0:
                value = self._getDefaultGamma(dset)
            kargs += [value]
        kernel = self._KERNELS[kernel_type_literal][0](sglhs, sgrhs, *kargs)
        
        # To be consistent with base SVM class, set normalizer to I
        # XXX: Why?  seems to help in some cases
        kernel.set_normalizer(IdentityKernelNormalizer())
        
        return kernel.get_kernel_matrix()
    
    def cacheNewLhsKernel(self, lhs, rhs, kernel_type_literal=None):
        """Caches the lhs vs rhs matrix, stores in __cached_lhs
        
        This differs from standard self.cache() as a rectangular matrix is 
        formed, so it is innappropriate for training data (the K(lh_i, lh_j)
        calculation is not present).  However, if the original lhs data is 
        resent each time, this is usefull to quickly recalculate the kernel
        for testing on new validation (ie rhs) data on an already-trained 
        classifier.  That is, the lhs data must be the original, uncached,
        data with all the same samples and features (and the same order) the 
        kernel was trained on.
        
        Useful if lhs is training data (noncached) and rhs is new test data
        
        eg: ctrain = self.cache(train); self.train(ctrain); 
            self.predict(self.cacheNewLhsKernel(train, test1))
            self.predict(self.cacheNewLhsKernel(train, test2))
            ...
            
        :Parameters:
          lhs: lhs dataset or samples (training set)
          rhs: rhs dataset or samples (testing set)
          kernel_type_literal: string specifying which kernel to use
          
        :Returns:
          rhscache: cached NDarray/dataset indeces.  The LHS is not returned 
          since this is presumed to be already cached
          
        """
        lhsd, ln = self._asdata(lhs)
        rhsd, rn = self._asdata(rhs)
        #lout = self._ascache(N.arange(ln), lhs)
        rout = self._ascache(N.arange(rn), rhs)
        
        clhs = self.calculateFullKernel(lhsd, rhs=rhsd, 
                                        kernel_type_literal=kernel_type_literal)
        self.__cached_lhs = clhs
        return rout
        
    def __makeKernelFromFull(self, full, train=False):
        """Creates a CustomKernel set with self.__cached_kernel
        if train, also calls IdentityKernelNormalizer (dunno if this does
        anything, but it's in SVM parent in training)
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
        m = self.__getCachedMatrix(dataset.samples)
        self._SVM__kernel = self.__makeKernelFromFull(m, train=True)
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
    
    def _predict(self, samples):
    
        # Builds the kernel from cached
        self.assertCachedData(samples)
        f = self.__getCachedRHS(samples)
        self._SVM__kernel_test = self.__makeKernelFromFull(f)
        
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
    
    The primary difference between this class and a CachedSVM with an rbf kernel
    is that this one will automagically update the kernel when gamma is changed,
    avoiding the need to recalculate it.
    
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
            msg = 'CachedRbfSVM must have kernel type ''rbf'''
            msg += ', not ''%s'''%kernel_type
            raise RuntimeError(msg)
        # Actually inits as a linear kernel, since we need the distance
        CachedSVM.__init__(self, kernel_type='linear', **kwargs)
        self.__distance = None # holder for cached distance matrix
        #self.gamma=1.
        
    def cache(self, d):
        """Caches the dataset, here in linear form to faciliate magic gamma
        changing"""
        dout = CachedSVM.cache(self, d)
        self.__distance = -self.__linearToDist(self._CachedSVM__cached_kernel)
        self.__updateCache()
        return dout
    
    def cacheNewLhsKernel(self, lhs, rhs=None, kernel_type_literal=None):
        """Creates a full kernel matrix from lhs and rhs
        
        This is a hack to allow this classifier to predict new data, since it
        won't normally work without caching the entire dataset
        
        Note that automatic gamma updating will not work for this caching method
        (since the full kernel is ignored here)
        """
        # Overrides because we need the rbf here, since we can't calculate the
        # distance between points from linear product without calculating whole
        # kernel matrix
        return CachedSVM.cacheNewLhsKernel(self, lhs, rhs, 
                                           kernel_type_literal='rbf')
        
    def __updateCache(self):
        """Creates the Rbf kernel from the linear distance kernel"""
        if not self.__distance is None:
            self._CachedSVM__cached_kernel = N.exp(self.__distance/self.gamma)
            
    @staticmethod
    def __linearToDist(Kij):
        """Convert a square dot product kernel matrix to Euclidean distance^2
        
        Formula:

        <x,y>=dot(x,y); |x| = sqrt(sum(x^2)); => |x|^2 = <x,x>
        
        |x-y|^2 = <x-y,x-y> = |x|^2 - 2<x,y> + |y|^2
        
        Therefor for kernel Kij = <xi, xj>, (Dij)^2 = Kii - 2Kij + Kjj
        """
        assert len(Kij.shape)==2
        assert Kij.shape[0] == Kij.shape[1]
        Kii = N.diag(Kij).reshape((Kij.shape[1], 1))
        Kjj = Kii.T
        return Kii - 2*Kij + Kjj
    
    def __setattr__(self, attr, value):
        """Automagically adjust kernel for new value of gamma
        otherwise set normally
        """
        CachedSVM.__setattr__(self, attr, value)
        if attr == 'gamma':
            self.__updateCache()
            
    def _getDefaultC(self, dataset):
        """Default C for the Rbf is 1"""
        #algorithm always returns 1.0 since diagonal is always 1.0 in Rbf
        # this also prevents some errors
        return 1.0
    
    def _getDefaultGamma(self, dataset):
        """Gets default gamma value
        
        :Parameters:
          dataset: Dataset instance on which to calculate default gamma
        """
        # XXX: Have to override default which checks for known parameters. 
        # maybe unify api at some point in future 
        value = 1.0 / len(dataset.uniquelabels)
        if __debug__:
            debug("SVM", "Default Gamma is computed to be %f" % value)
            
        return value

