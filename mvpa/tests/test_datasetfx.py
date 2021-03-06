# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA miscelaneouse functions operating on datasets"""

import unittest
from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_equal

import numpy as N

from mvpa.base import externals
from mvpa.datasets.base import dataset
from mvpa.datasets.miscfx import removeInvariantFeatures, coarsenChunks, \
        aggregateFeatures, zscore


from mvpa.misc.data_generators import normalFeatureDataset

class MiscDatasetFxTests(unittest.TestCase):

    def testAggregation(self):
        data = dataset(N.arange( 20 ).reshape((4, 5)), labels=1, chunks=1)

        ag_data = aggregateFeatures(data, N.mean)

        ok_(ag_data.nsamples == 4)
        ok_(ag_data.nfeatures == 1)
        assert_array_equal(ag_data.samples[:, 0], [2, 7, 12, 17])


    def testInvarFeaturesRemoval(self):
        r = N.random.normal(size=(3,1))
        ds = dataset(samples=N.hstack((N.zeros((3,2)), r)),
                     labels=1)

        self.failUnless(ds.nfeatures == 3)

        dsc = removeInvariantFeatures(ds)

        self.failUnless(dsc.nfeatures == 1)
        self.failUnless((dsc.samples == r).all())


    def testCoarsenChunks(self):
        """Just basic testing for now"""
        chunks = [1,1,2,2,3,3,4,4]
        ds = dataset(samples=N.arange(len(chunks)).reshape(
            (len(chunks),1)), labels=[1]*8, chunks=chunks)
        coarsenChunks(ds, nchunks=2)
        chunks1 = coarsenChunks(chunks, nchunks=2)
        print type(chunks1)
        self.failUnless((chunks1 == ds.chunks).all())
        self.failUnless((chunks1 == N.asarray([0,0,0,0,1,1,1,1])).all())

        ds2 = dataset(samples=N.arange(len(chunks)).reshape(
            (len(chunks),1)), labels=[1]*8, chunks=range(len(chunks)))
        coarsenChunks(ds2, nchunks=2)
        self.failUnless((chunks1 == ds.chunks).all())

    def testBinds(self):
        ds = normalFeatureDataset()
        ds_data = ds.samples.copy()
        ds_chunks = ds.chunks.copy()
        self.failUnless(N.all(ds.samples == ds_data)) # sanity check

        funcs = ['zscore', 'coarsenChunks']
        if externals.exists('scipy'):
            funcs.append('detrend')

        for f in funcs:
            eval('ds.%s()' % f)
            self.failUnless(N.any(ds.samples != ds_data) or
                            N.any(ds.chunks != ds_chunks),
                msg="We should have modified original dataset with %s" % f)
            ds.samples = ds_data.copy()
            ds.sa['chunks'].value = ds_chunks.copy()

        # and some which should just return results
        for f in ['aggregateFeatures', 'removeInvariantFeatures',
                  'getSamplesPerChunkLabel']:
            res = eval('ds.%s()' % f)
            self.failUnless(res is not None,
                msg='We should have got result from function %s' % f)
            self.failUnless(N.all(ds.samples == ds_data),
                msg="Function %s should have not modified original dataset" % f)


def test_zscoring():
    """Test z-scoring transformation
    """
    # dataset: mean=2, std=1
    samples = N.array((0, 1, 3, 4, 2, 2, 3, 1, 1, 3, 3, 1, 2, 2, 2, 2)).\
        reshape((16, 1))
    data = dataset(samples.copy(), labels=range(16), chunks=[0] * 16)
    assert_equal(data.samples.mean(), 2.0)
    assert_equal(data.samples.std(), 1.0)
    zscore(data, perchunk=True)

    # check z-scoring
    check = N.array([-2, -1, 1, 2, 0, 0, 1, -1, -1, 1, 1, -1, 0, 0, 0, 0],
                    dtype='float64').reshape(16, 1)
    assert_array_equal(data.samples, check)

    data = dataset(samples.copy(), labels=range(16), chunks=[0] * 16)
    zscore(data, perchunk=False)
    assert_array_equal(data.samples, check)

    # check z-scoring taking set of labels as a baseline
    data = dataset(samples.copy(),
                   labels=[0, 2, 2, 2, 1] + [2] * 11,
                   chunks=[0] * 16)
    zscore(data, baselinelabels=[0, 1])
    assert_array_equal(samples, data.samples + 1.0)

    # check that zscore modifies in-place; only guaranteed if no upcasting is
    # necessary
    samples = samples.astype('float')
    data = dataset(samples,
                   labels=[0, 2, 2, 2, 1] + [2] * 11,
                   chunks=[0] * 16)
    zscore(data, baselinelabels=[0, 1])
    assert_array_equal(samples, data.samples)





def suite():
    return unittest.makeSuite(MiscDatasetFxTests)


if __name__ == '__main__':
    import runner

