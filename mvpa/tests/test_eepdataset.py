# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA EEP dataset"""

import unittest
import os.path
import numpy as N

from mvpa import pymvpa_dataroot
from mvpa.base import externals
from mvpa.datasets.eep import eep_dataset
from mvpa.misc.io.eepbin import EEPBin


class EEPDatasetTests(unittest.TestCase):

    def testLoad(self):
        eb = EEPBin(os.path.join(pymvpa_dataroot, 'eep.bin'))

        ds = [ eep_dataset(source, labels=[1, 2]) for source in
                (eb, os.path.join(pymvpa_dataroot, 'eep.bin')) ]

        for d in ds:
            self.failUnless(d.nsamples == 2)
            self.failUnless(d.nfeatures == 128)
            self.failUnless(d.a.channelids[23] == 'Pz')
            self.failUnless(N.round(d.a.t0 + 0.002, decimals=3) == 0)
            self.failUnless(N.round(d.a.dt - 0.002, decimals=3) == 0)
            self.failUnless(N.round(d.samplingrate) == 500)


    def testEEPBin(self):
        eb = EEPBin(os.path.join(pymvpa_dataroot, 'eep.bin'))

        self.failUnless(eb.nchannels == 32)
        self.failUnless(eb.nsamples == 2)
        self.failUnless(eb.ntimepoints == 4)
        self.failUnless(eb.t0 - eb.dt < 0.00000001)
        self.failUnless(len(eb.channels) == 32)
        self.failUnless(eb.data.shape == (2, 32, 4))


    # XXX put me back whenever there is a proper resamples()
#     def testResampling(self):
#         ds = eep_dataset(os.path.join(pymvpa_dataroot, 'eep.bin'),
#                          labels=[1, 2])
#         channelids = N.array(ds.a.channelids).copy()
#         self.failUnless(N.round(ds.samplingrate) == 500.0)
# 
#         if not externals.exists('scipy'):
#             return
# 
#         # should puke when called with nothing
#         self.failUnlessRaises(ValueError, ds.resample)
# 
#         # now for real -- should divide nsamples into half
#         rds = ds.resample(sr=250, inplace=False)
#         # We should have not changed anything
#         self.failUnless(N.round(ds.samplingrate) == 500.0)
# 
#         # by default do 'inplace' resampling
#         ds.resample(sr=250)
#         for d in [rds, ds]:
#             self.failUnless(N.round(d.samplingrate) == 250)
#             self.failUnless(d.nsamples == 2)
#             self.failUnless(N.abs((d.a.dt - 1.0/250)/d.a.dt)<1e-5)
#             self.failUnless(N.all(d.a.channelids == channelids))
#             # lets now see if we still have a mapper
#             self.failUnless(d.O.shape == (2, len(channelids), 2))

def suite():
    return unittest.makeSuite(EEPDatasetTests)


if __name__ == '__main__':
    import runner

