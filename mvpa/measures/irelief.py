# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Emanuele Olivetti <emanuele@relativita.com>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""FeaturewiseDatasetMeasure performing multivariate Iterative RELIEF
(I-RELIEF) algorithm.
See : Y. Sun, Iterative RELIEF for Feature Weighting: Algorithms, Theories,
and Applications, IEEE Trans. on Pattern Analysis and Machine Intelligence
(TPAMI), vol. 29, no. 6, pp. 1035-1051, June 2007."""


__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.measures.base import FeaturewiseDatasetMeasure
from mvpa.clfs.kernel import KernelSquaredExponential, KernelExponential, \
     KernelMatern_3_2, KernelMatern_5_2
from mvpa.clfs.distance import pnorm_w

if __debug__:
    from mvpa.base import debug


class IterativeRelief_Devel(FeaturewiseDatasetMeasure):
    """`FeaturewiseDatasetMeasure` that performs multivariate I-RELIEF
    algorithm. Batch version allowing various kernels.

    UNDER DEVELOPEMNT.

    Batch I-RELIEF-2 feature weighting algorithm. Works for binary or
    multiclass class-labels. Batch version with complexity O(T*N^2*I),
    where T is the number of iterations, N the number of instances, I
    the number of features.

    See: Y. Sun, Iterative RELIEF for Feature Weighting: Algorithms,
    Theories, and Applications, IEEE Trans. on Pattern Analysis and
    Machine Intelligence (TPAMI), vol. 29, no. 6, pp. 1035-1051, June
    2007. http://plaza.ufl.edu/sunyijun/Paper/PAMI_1.pdf

    Note that current implementation allows to use only
    exponential-like kernels. Support for linear kernel will be
    added later.
    """
    def __init__(self, threshold = 1.0e-2, kernel = None, kernel_width = 1.0,
                 w_guess = None, **kwargs):
        """Constructor of the IRELIEF class.

        """
        # init base classes first
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)

        # Threshold in W changes (stopping criterion for irelief)
        self.threshold = threshold
        if kernel == None:
            self.kernel = KernelExponential
        else:
            self.kernel = kernel

        self.w_guess = w_guess
        self.w = None
        self.kernel_width = kernel_width


    def compute_M_H(self, label):
        """Compute hit/miss dictionaries.

        For each instance compute the set of indices having the same
        class label and different class label.

        Note that this computation is independent of the number of
        features.
        """

        M = {}
        H = {}
        for i in range(label.size):
            M[i] = N.where(label != label[i])[0]
            tmp = (N.where(label == label[i])[0]).tolist()
            tmp.remove(i)
            # There must be at least two exampls for class label[i]
            assert(tmp != [])
            H[i] = N.array(tmp)

        return M, H


    def _call(self, dataset):
        """Computes featurewise I-RELIEF weights."""
        samples = dataset.samples
        NS, NF = samples.shape[:2]
        if self.w_guess == None:
            self.w = N.ones(NF, 'd')
        # do normalization in all cases to be safe :)
        self.w = self.w/(self.w**2).sum()

        M, H = self.compute_M_H(dataset.labels)

        while True:
            self.k = self.kernel(length_scale = self.kernel_width/self.w)
            d_w_k = self.k.compute(samples)
            # set d_w_k to zero where distance=0 (i.e. kernel ==
            # 1.0), otherwise I-RELIEF could not converge.
            # XXX Note that kernel==1 for distance=0 only for
            # exponential kernels!!  IMPROVE
            d_w_k[N.abs(d_w_k-1.0) < 1.0e-15] = 0.0
            ni = N.zeros(NF, 'd')
            for n in range(NS):
                # d_w_k[n,n] could be omitted since == 0.0
                gamma_n = 1.0 - N.nan_to_num(d_w_k[n, M[n]].sum() \
                                / (d_w_k[n, :].sum()-d_w_k[n, n]))
                alpha_n = N.nan_to_num(d_w_k[n, M[n]]/(d_w_k[n, M[n]].sum()))
                beta_n = N.nan_to_num(d_w_k[n, H[n]]/(d_w_k[n, H[n]].sum()))

                m_n = (N.abs(samples[n, :] - samples[M[n], :]) \
                        * alpha_n[:, None]).sum(0)
                h_n = (N.abs(samples[n, :] - samples[H[n], :]) \
                        * beta_n[:, None]).sum(0)
                ni += gamma_n*(m_n-h_n)
            ni = ni/NS

            ni_plus = N.clip(ni, 0.0, N.inf) # set all negative elements to zero
            w_new = N.nan_to_num(ni_plus/(N.sqrt((ni_plus**2).sum())))
            change = N.abs(w_new-self.w).sum()
            if __debug__ and 'IRELIEF' in debug.active:
                debug('IRELIEF',
                      "change=%.4f max=%f min=%.4f mean=%.4f std=%.4f #nan=%d"
                      % (change, w_new.max(), w_new.min(), w_new.mean(),
                         w_new.std(), N.isnan(w_new).sum()))

            # update weights:
            self.w = w_new
            if change < self.threshold:
                break

        return self.w


class IterativeReliefOnline_Devel(IterativeRelief_Devel):
    """`FeaturewiseDatasetMeasure` that performs multivariate I-RELIEF
    algorithm. Online version.

    UNDER DEVELOPMENT

    Online version with complexity O(T*N*I),
    where N is the number of instances and I the number of features.

    See: Y. Sun, Iterative RELIEF for Feature Weighting: Algorithms,
    Theories, and Applications, IEEE Trans. on Pattern Analysis and
    Machine Intelligence (TPAMI), vol. 29, no. 6, pp. 1035-1051, June
    2007. http://plaza.ufl.edu/sunyijun/Paper/PAMI_1.pdf

    Note that this implementation is not fully online, since hit and
    miss dictionaries (H,M) are computed once at the beginning using
    full access to all labels. This can be easily corrected to a full
    online implementation. But this is not mandatory now since the
    major goal of this current online implementation is reduction of
    computational complexity.
    """

    def __init__(self, a=5.0, permute=True, max_iter=3, **kwargs):
        """Constructor of the IRELIEF class.

        """
        # init base classes first
        IterativeRelief_Devel.__init__(self, **kwargs)

        self.a = a # parameter of the learning rate
        self.permute = permute # shuffle data when running I-RELIEF
        self.max_iter = max_iter # maximum number of iterations


    def _call(self, dataset):
        """Computes featurewise I-RELIEF-2 weights. Online version."""
        NS = dataset.samples.shape[0]
        NF = dataset.samples.shape[1]
        if self.w_guess == None:
            self.w = N.ones(NF, 'd')
        # do normalization in all cases to be safe :)
        self.w = self.w/(self.w**2).sum()

        M, H = self.compute_M_H(dataset.labels)

        ni = N.zeros(NF, 'd')
        pi = N.zeros(NF, 'd')

        if self.permute:
            # indices to go through samples in random order
            random_sequence = N.random.permutation(NS)
        else:
            random_sequence = N.arange(NS)

        change = self.threshold + 1.0
        iteration = 0
        counter = 0.0
        while change > self.threshold and iteration < self.max_iter:
            if __debug__:
                debug('IRELIEF', "Iteration %d" % iteration)

            for t in range(NS):
                counter += 1.0
                n = random_sequence[t]

                self.k = self.kernel(length_scale = self.kernel_width/self.w)
                d_w_k_xn_Mn = self.k.compute(dataset.samples[None, n, :],
                                dataset.samples[M[n], :]).squeeze()
                d_w_k_xn_Mn_sum = d_w_k_xn_Mn.sum()
                d_w_k_xn_x = self.k.compute(dataset.samples[None, n, :],
                                dataset.samples).squeeze()
                gamma_n = 1.0 - d_w_k_xn_Mn_sum / d_w_k_xn_x.sum()
                alpha_n = d_w_k_xn_Mn / d_w_k_xn_Mn_sum

                d_w_k_xn_Hn = self.k.compute(dataset.samples[None, n, :],
                                dataset.samples[H[n], :]).squeeze()
                beta_n = d_w_k_xn_Hn / d_w_k_xn_Hn.sum()

                m_n = (N.abs(dataset.samples[n, :] - dataset.samples[M[n], :]) \
                        * alpha_n[:, N.newaxis]).sum(0)
                h_n = (N.abs(dataset.samples[n, :] - dataset.samples[H[n], :]) \
                        * beta_n[:, N.newaxis]).sum(0)
                pi = gamma_n * (m_n-h_n)
                learning_rate = 1.0 / (counter * self.a + 1.0)
                ni_new = ni + learning_rate * (pi - ni)
                ni = ni_new

                # set all negative elements to zero
                ni_plus = N.clip(ni, 0.0, N.inf)
                w_new = N.nan_to_num(ni_plus / (N.sqrt((ni_plus ** 2).sum())))
                change = N.abs(w_new - self.w).sum()
                if t % 10 == 0 and __debug__ and 'IRELIEF' in debug.active:
                    debug('IRELIEF',
                          "t=%d change=%.4f max=%f min=%.4f mean=%.4f std=%.4f"
                          " #nan=%d" %
                          (t, change, w_new.max(), w_new.min(), w_new.mean(),
                           w_new.std(), N.isnan(w_new).sum()))

                self.w = w_new

                if change < self.threshold and iteration > 0:
                    break

            iteration += 1

        return self.w



class IterativeRelief(FeaturewiseDatasetMeasure):
    """`FeaturewiseDatasetMeasure` that performs multivariate I-RELIEF
    algorithm. Batch version.

    Batch I-RELIEF-2 feature weighting algorithm. Works for binary or
    multiclass class-labels. Batch version with complexity O(T*N^2*I),
    where T is the number of iterations, N the number of instances, I
    the number of features.

    See: Y. Sun, Iterative RELIEF for Feature Weighting: Algorithms,
    Theories, and Applications, IEEE Trans. on Pattern Analysis and
    Machine Intelligence (TPAMI), vol. 29, no. 6, pp. 1035-1051, June
    2007. http://plaza.ufl.edu/sunyijun/Paper/PAMI_1.pdf

    Note that current implementation allows to use only
    exponential-like kernels. Support for linear kernel will be
    added later.
    """
    def __init__(self, threshold=1.0e-2, kernel_width=1.0,
                 w_guess=None, **kwargs):
        """Constructor of the IRELIEF class.

        """
        # init base classes first
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)

        # Threshold in W changes (stopping criterion for irelief).
        self.threshold = threshold
        self.w_guess = w_guess
        self.w = None
        self.kernel_width = kernel_width


    def compute_M_H(self, label):
        """Compute hit/miss dictionaries.

        For each instance compute the set of indices having the same
        class label and different class label.

        Note that this computation is independent of the number of
        features.

        XXX should it be some generic function since it doesn't use self
        """

        M = {}
        H = {}
        for i in range(label.size):
            M[i] = N.where(label != label[i])[0]
            tmp = (N.where(label == label[i])[0]).tolist()
            tmp.remove(i)
            # There must be least two exampls for class label[i]
            assert(tmp != [])
            H[i] = N.array(tmp)

        return M, H


    def k(self, distances):
        """Exponential kernel."""
        kd = N.exp(-distances/self.kernel_width)
        # set kd to zero where distance=0 otherwise I-RELIEF could not converge.
        kd[N.abs(distances) < 1.0e-15] = 0.0
        return kd


    def _call(self, dataset):
        """Computes featurewise I-RELIEF weights."""
        samples = dataset.samples
        NS, NF = samples.shape[:2]

        if self.w_guess == None:
            w = N.ones(NF, 'd')

        w /= (w ** 2).sum() # do normalization in all cases to be safe :)

        M, H = self.compute_M_H(dataset.labels)

        while True:
            d_w_k = self.k(pnorm_w(data1=samples, weight=w, p=1))
            ni = N.zeros(NF, 'd')
            for n in range(NS):
                 # d_w_k[n, n] could be omitted since == 0.0
                gamma_n = 1.0 - N.nan_to_num(d_w_k[n, M[n]].sum() \
                                / (d_w_k[n, :].sum() - d_w_k[n, n]))
                alpha_n = N.nan_to_num(d_w_k[n, M[n]] / (d_w_k[n, M[n]].sum()))
                beta_n = N.nan_to_num(d_w_k[n, H[n]] / (d_w_k[n, H[n]].sum()))

                m_n = (N.abs(samples[n, :] - samples[M[n], :]) \
                       * alpha_n[:, None]).sum(0)
                h_n = (N.abs(samples[n, :] - samples[H[n], :]) \
                       * beta_n[:, None]).sum(0)
                ni += gamma_n*(m_n - h_n)

            ni = ni / NS

            ni_plus = N.clip(ni, 0.0, N.inf) # set all negative elements to zero
            w_new = N.nan_to_num(ni_plus / (N.sqrt((ni_plus**2).sum())))
            change = N.abs(w_new - w).sum()
            if __debug__ and 'IRELIEF' in debug.active:
                debug('IRELIEF',
                      "change=%.4f max=%f min=%.4f mean=%.4f std=%.4f #nan=%d" \
                      % (change, w_new.max(), w_new.min(), w_new.mean(),
                         w_new.std(), N.isnan(w_new).sum()))

            # update weights:
            w = w_new
            if change < self.threshold:
                break

        self.w = w
        return w


class IterativeReliefOnline(IterativeRelief):
    """`FeaturewiseDatasetMeasure` that performs multivariate I-RELIEF
    algorithm. Online version.

    This algorithm is exactly the one in the referenced paper
    (I-RELIEF-2 online), using weighted 1-norm and Exponential
    Kernel.
    """

    def __init__(self, a=10.0, permute=True, max_iter=3, **kwargs):
        """Constructor of the IRELIEF class.

        """
        # init base classes first
        IterativeRelief.__init__(self, **kwargs)

        self.a = a # parameter of the learning rate
        self.permute = permute # shuffle data when running I-RELIEF
        self.max_iter = max_iter # maximum number of iterations


    def _call(self, dataset):
        """Computes featurewise I-RELIEF-2 weights. Online version."""
        # local bindings
        samples = dataset.samples
        NS, NF = samples.shape[:2]
        threshold = self.threshold
        a = self.a

        if self.w_guess == None:
            w = N.ones(NF, 'd')

        # do normalization in all cases to be safe :)
        w /= (w ** 2).sum()

        M, H = self.compute_M_H(dataset.labels)

        ni = N.zeros(NF, 'd')
        pi = N.zeros(NF, 'd')

        if self.permute:
            # indices to go through x in random order
            random_sequence = N.random.permutation(NS)
        else:
            random_sequence = N.arange(NS)

        change = threshold + 1.0
        iteration = 0
        counter = 0.0
        while change > threshold and iteration < self.max_iter:
            if __debug__:
                debug('IRELIEF', "Iteration %d" % iteration)
            for t in range(NS):
                counter += 1.0
                n = random_sequence[t]

                d_xn_x = N.abs(samples[n, :] - samples)
                d_w_k_xn_x = self.k((d_xn_x * w).sum(1))

                d_w_k_xn_Mn = d_w_k_xn_x[M[n]]
                d_w_k_xn_Mn_sum = d_w_k_xn_Mn.sum()

                gamma_n = 1.0 - d_w_k_xn_Mn_sum / d_w_k_xn_x.sum()
                alpha_n = d_w_k_xn_Mn / d_w_k_xn_Mn_sum

                d_w_k_xn_Hn = d_w_k_xn_x[H[n]]
                beta_n = d_w_k_xn_Hn / d_w_k_xn_Hn.sum()

                m_n = (d_xn_x[M[n], :] * alpha_n[:, None]).sum(0)
                h_n = (d_xn_x[H[n], :] * beta_n[:, None]).sum(0)
                pi = gamma_n * (m_n - h_n)
                learning_rate = 1.0 / (counter * a + 1.0)
                ni_new = ni + learning_rate * (pi - ni)
                ni = ni_new

                # set all negative elements to zero
                ni_plus = N.clip(ni, 0.0, N.inf)
                w_new = N.nan_to_num(ni_plus / (N.sqrt((ni_plus ** 2).sum())))
                change = N.abs(w_new - w).sum()
                if t % 10 == 0 and __debug__ and 'IRELIEF' in debug.active:
                    debug('IRELIEF',
                          "t=%d change=%.4f max=%f min=%.4f mean=%.4f std=%.4f"
                          " #nan=%d" %
                          (t, change, w_new.max(), w_new.min(), w_new.mean(),
                           w_new.std(), N.isnan(w_new).sum()))

                w = w_new

                if change < threshold and iteration > 0:
                    break

            iteration += 1

        self.w = w
        return w

