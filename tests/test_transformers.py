#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA transformers."""

import unittest
import numpy as N

from mvpa.misc.transformers import Absolute, OneMinus


class TransformerTests(unittest.TestCase):

    def testAbsolute(self):
        functor = N.random.normal

        abs_func = Absolute(functor)

        # generate 100 values (gaussian noise mean -1000 -> all negative)
        out = abs_func(-1000, size=100)

        self.failUnless(out.min() >= 0)
        self.failUnless(len(out) == 100)


    def testAbsolute(self):
        functor = N.arange
        target = N.array([ 1,  0, -1, -2, -3])

        om_func = OneMinus(functor)

        out = om_func(5)
        self.failUnless((out == target).all())


def suite():
    return unittest.makeSuite(TransformerTests)


if __name__ == '__main__':
    import test_runner
