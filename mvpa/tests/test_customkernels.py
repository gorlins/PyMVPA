from tests_warehouse import *
from mvpa import externals

"""Tests that the custom cached kernels behave identically to their non-cached
counterparts

My first unittest ever, so could use some work
Todo:
* learn to use unittests to sweep through various (cached) classifiers
 (currently they're all in same tests)
* break up framework into multiple smaller tests instead of one big one
"""
shogun = externals.exists('shogun')

if shogun:
    import mvpa.clfs.sg.custom as shogun_custom
class CustomKernelTest(unittest.TestCase):

    def testCustomKernels(self):
        self.dset = datasets['dumb2']
        
        self.failUnless(shogun, msg='Custom (cached) kernel classifiers ' + 
                        'require Shogun')
        
        try:
            self.sg_csvm = shogun_custom.CachedSVM()
            self.sg_crbf = shogun_custom.CachedRbfSVM()
        except Exception:
            self.fail(msg='Could not create classifiers')
        try:
            self.sg_svm = shogun_custom.SVM()
            self.sg_rbf = shogun_custom.SVM(kernel_type='rbf')
        except Exception:
            self.fail(msg='Could not create standard Shogun classifiers')
        
        #def testKernelSetParams(self):
        try:
            self.sg_svm.C = .1
            self.sg_csvm.C = .1
            self.sg_rbf.C = 0.1
            self.sg_crbf.C = 0.1
            self.sg_crbf.gamma = 1.
            self.sg_rbf.gamma = 1.
        except Exception:
            self.fail(msg='Problem setting classifier (kernel) parameters')
    
        #def testAssertCached(self):
        try:
            self.sg_csvm.assertCachedData(self.dset)
        except AssertionError:
            pass #failUnlessRaises not working for me...
        except Exception:
            self.fail(msg='Problem detecting non-cached data')
        
        #def testKernelCache(self):
        try:
            self.cached_linear = self.sg_csvm.cache(self.dset)
            self.cached_rbf = self.sg_crbf.cache(self.dset)
        except Exception:
            self.fail(msg='Problem caching data')
        try:
            self.sg_csvm.assertCachedData(self.cached_linear)
            self.sg_crbf.assertCachedData(self.cached_rbf)
        except AssertionError:
            self.fail('Problem asserting correctly cached data')
            
        # Handle default C (if data is cached)
        cnormal = self.sg_svm._getDefaultC(self.dset.samples)
        ccached = self.sg_csvm._getDefaultC(self.cached_linear.samples)
        self.failUnless(cnormal==ccached, msg='Problem calculating C')
        
        #def testTraining(self):
        try:
            self.sg_csvm.train(self.cached_linear)
            self.sg_crbf.train(self.cached_rbf)
        except Exception:
            self.fail(msg='Problem training classifiers')
            
        try:
            self.sg_svm.train(self.dset)
            self.sg_rbf.train(self.dset)
        except Exception:
            self.fail(msg='Problem training standard Shogun classifiers')
        
        
        #def testAlphaIdentities(self):
        self.failUnless((self.sg_csvm._SVM__svm.get_alphas() == 
                         self.sg_svm._SVM__svm.get_alphas()).all(),
                        msg='CachedSVM did not calculate alpha properly')
        self.failUnless((self.sg_crbf._SVM__svm.get_alphas() ==
                         self.sg_rbf._SVM__svm.get_alphas()).all(),
                        msg='CachedRbfSVM did not calculate alpha properly')
        
        #def testChangeGamma(self):
        self.sg_crbf.gamma = .5 # Updates kernel w/o recalculating :)!!
        self.sg_rbf.gamma=.5
        self.sg_rbf.train(self.dset)
        self.failUnless((self.sg_crbf._ck._cached_kernel == 
                         self.sg_rbf._SVM__kernel.get_kernel_matrix()).all(),
                        msg='CachedRbfSVM did not update gamma properly')
        
        #def testRecacheClassify(self):
        (d1, d2) = self.sg_csvm.cacheMultiple(self.dset, self.dset)
        self.sg_csvm.predict(d2.samples)
        self.sg_svm.predict(self.dset.samples)
        self.failUnless((self.sg_csvm.values == self.sg_svm.values).all(),
                        msg='CachedSVM failed to predict values properly')
        
        (d1, d2) = self.sg_crbf.cacheMultiple(self.dset, self.dset)
        self.sg_crbf.predict(d2.samples)
        self.sg_rbf.predict(self.dset.samples)
        self.failUnless((N.abs(self.sg_crbf.values - 
                               self.sg_rbf.values)<1e-8).all(),
                        msg='CachedRbfSVM failed to predict values properly')
        #On occaision, the diffs test at ~=4e-10 on my computer

        # def testUncachedTestData(self):
        td = self.sg_csvm.cacheNewLhsKernel(self.dset, self.dset)
        self.sg_csvm.predict(td.samples)
        self.failUnless((self.sg_csvm.values == self.sg_svm.values).all(),
                        msg='CachedSVM failed to predict values properly')
        
        td = self.sg_crbf.cacheNewLhsKernel(self.dset, self.dset)
        self.sg_crbf.predict(td.samples)
        self.failUnless((N.abs(self.sg_crbf.values - 
                               self.sg_rbf.values)<1e-8).all(),
                        msg='CachedRbfSVM failed to predict values properly')
        #For some reason, the diffs test at ~=4e-10 on my computer


def suite():
    return unittest.makeSuite(CustomKernelTest)

def run():
    import runner
if __name__ == '__main__':
    run()
