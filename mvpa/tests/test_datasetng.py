'''Tests for the dataset implementation'''
import numpy as N

from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises

from mvpa.datasets.base import Dataset
from mvpa.mappers.array import DenseArrayMapper
from mvpa.misc.data_generators import normalFeatureDataset
import mvpa.support.copy as copy
from mvpa.misc.exceptions import DatasetError

def test_from_labeled():
    samples = N.arange(12).reshape((4,3))
    labels = range(4)
    chunks = [1, 1, 2, 2]

    ds = Dataset.from_labeled(samples, labels, chunks)

    ## XXX stuff that needs thought:

    # ds.reset() << I guess we don't want that, but it is there.

    # ds.sa (empty) has this in the public namespace:
    #   add, get, getvalue, isKnown, isSet, items, listing, name, names
    #   owner, remove, reset, setvalue, whichSet
    # maybe we need some form of leightweightCollection?

    assert_array_equal(ds.samples, samples)
    assert_array_equal(ds.sa.labels, labels)
    assert_array_equal(ds.sa.chunks, chunks)

    # same should work for shortcuts
    assert_array_equal(ds.labels, labels)
    assert_array_equal(ds.chunks, chunks)

    ok_(sorted(ds.sa.names) == ['chunks', 'labels'])

    # there is not necessarily a mapper present
    ok_(not ds.a.isKnown('mapper'))

    # has to complain about misshaped samples attributes
    assert_raises(DatasetError, Dataset.from_labeled, samples, labels + labels)


def test_basic_datamapping():
    samples = N.arange(24).reshape((4,3,2))

    # cannot handle 3d samples without a mapper
    assert_raises(DatasetError, Dataset, samples)

    ds = Dataset.from_unlabeled(samples,
            mapper=DenseArrayMapper(shape=samples.shape[1:]))

    # mapper should end up in the dataset
    ok_(ds.a.isKnown('mapper') == ds.a.isSet('mapper') == True)

    # check correct mapping
    ok_(ds.nsamples == 4)
    ok_(ds.nfeatures == 6)


def test_ds_copy():
    # lets use some instance of somewhat evolved dataset
    ds = normalFeatureDataset()
    # Clone the beast
    ds_ = copy.deepcopy(ds)
    # verify that we have the same data
    assert_array_equal(ds.samples, ds_.samples)
    assert_array_equal(ds.labels, ds_.labels)
    assert_array_equal(ds.chunks, ds_.chunks)

    # modify and see if we don't change data in the original one
    ds_.samples[0, 0] = 1234
    ok_(N.any(ds.samples != ds_.samples))
    assert_array_equal(ds.labels, ds_.labels)
    assert_array_equal(ds.chunks, ds_.chunks)

    ds_.sa.labels = N.hstack(([123], ds_.labels[1:]))
    ok_(N.any(ds.samples != ds_.samples))
    ok_(N.any(ds.labels != ds_.labels))
    assert_array_equal(ds.chunks, ds_.chunks)

    ds_.sa.chunks = N.hstack(([1234], ds_.chunks[1:]))
    ok_(N.any(ds.samples != ds_.samples))
    ok_(N.any(ds.labels != ds_.labels))
    ok_(N.any(ds.chunks != ds_.chunks))

    # XXX implement me
    #ok_(N.any(ds.uniquelabels != ds_.uniquelabels))
    #ok_(N.any(ds.uniquechunks != ds_.uniquechunks))


def test_mergeds():
    data0 = Dataset.from_labeled(N.ones((5, 5)), labels=1)
    data1 = Dataset.from_labeled(N.ones((5, 5)), labels=1, chunks=1)
    data2 = Dataset.from_labeled(N.ones((3, 5)), labels=2, chunks=1)

    # cannot merge if there are attributes missing in one of the datasets
    assert_raises(DatasetError, data1.__iadd__, data0)

    merged = data1 + data2

    ok_( merged.nfeatures == 5 )
    l12 = [1]*5 + [2]*3
    l1 = [1]*8
    ok_((merged.labels == l12).all())
    ok_((merged.chunks == l1).all())

    data1 += data2

    ok_(data1.nfeatures == 5)
    ok_((data1.labels == l12).all())
    ok_((data1.chunks == l1).all())