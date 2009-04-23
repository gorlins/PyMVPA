from tests_warehouse import *
from mvpa import externals

"""Tests that the custom cached kernels behave identically to their non-cached
counterparts

My first unittest ever, so could use some work
Todo:
* learn to use unittests to sweep through various (cached) classifiers (currently
they're all in same tests)
* break up framework into multiple smaller tests instead of one big one
"""
shogun = externals.exists('shogun')

if shogun:
    import mvpa.clfs.sg.custom as shogun_custom
class CustomKernelTest(unittest.TestCase):

    def testCustomKernels(self):
        self.dset = datasets['dumb2']
        
        self.failUnless(shogun, msg='Custom (cached) kernel classifiers currently require Shogun')
        
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
    
        #def testKernelCache(self):
        try:
            self.cached_linear = self.sg_csvm.cache(self.dset)
            self.cached_rbf = self.sg_crbf.cache(self.dset)
        except Exception:
            self.fail(msg='Problem caching data')
        
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
        
        #def testKernelIdentities(self):
        self.failUnless((self.sg_csvm._CachedSVM__cached_kernel == self.sg_svm._SVM__kernel.get_kernel_matrix()).all(),
                        msg='CachedSVM did not calculate the kernel matrix properly')
        self.failUnless((self.sg_crbf._CachedSVM__cached_kernel == self.sg_rbf._SVM__kernel.get_kernel_matrix()).all(),
                        msg='CachedRbfSVM did not calculate the kernel matrix properly')
        
        #def testAlphaIdentities(self):
        self.failUnless((self.sg_csvm._CachedSVM__svm.get_alphas() == self.sg_svm._SVM__svm.get_alphas()).all(),
                        msg='CachedSVM did not calculate the support vectors properly')
        self.failUnless((self.sg_crbf._CachedSVM__svm.get_alphas() == self.sg_rbf._SVM__svm.get_alphas()).all(),
                        msg='CachedRbfSVM did not calculate the support vectors properly')
        
        #def testChangeGamma(self):
        self.sg_crbf.gamma = .5 # Automatically updates kernel without recalculating!!
        self.sg_rbf.gamma=.5
        self.sg_rbf.train(self.dset)
        self.failUnless((self.sg_crbf._CachedSVM__cached_kernel == self.sg_rbf._SVM__kernel.get_kernel_matrix()).all(),
                        msg='CachedRbfSVM did not update gamma properly')


def suite():
    return unittest.makeSuite(CustomKernelTest)


if __name__ == '__main__':
    import runner

