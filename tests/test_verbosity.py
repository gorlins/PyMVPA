#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA verbose and debug output"""

import unittest
from StringIO import StringIO

from mvpa.misc import verbose
if __debug__:
    from mvpa.misc import debug

## XXX There must be smth analogous in python... don't know it yet
# And it is StringIO
#class StringStream(object):
#    def __init__(self):
#        self.__str = ""
#
#    def __repr__(self):
#        return self.__str
#
#    def write(self, s):
#        self.__str += s
#
#    def clean(self):
#        self.__str = ""
#
class VerboseOutputTest(unittest.TestCase):

    def setUp(self):
        self.msg = "Test level 2"
        # output stream
        self.sout = StringIO()

        # set verbose to 4th level
        verbose.handlers = [self.sout]
        debug.handlers = [self.sout]

        verbose.level = 4
        debug.active = [1, 2, 'SLC']

    def tearDown(self):
        self.sout.close()

    def testVerboseAbove(self):
        """Test if it doesn't output at higher levels"""
        verbose(5, self.msg)
        self.failUnlessEqual(self.sout.getvalue(), "")

    def testVerboseBelow(self):
        """Test if outputs at lower levels and indents
        by default with spaces
        """
        verbose(2, self.msg)
        self.failUnlessEqual(self.sout.getvalue(),
                             "  %s\n" % self.msg)

    def testVerboseIndent(self):
        """Test indent symbol
        """
        verbose.indent = "."
        verbose(2, self.msg)
        self.failUnlessEqual(self.sout.getvalue(), "..%s\n" % self.msg)
        verbose.indent = " "            # restore

    def testVerboseNegative(self):
        """Test if chokes on negative level"""
        self.failUnlessRaises( ValueError,
                               verbose._setLevel, -10 )

    def testNoLF(self):
        """Test if it works fine with no newline (LF) symbol"""
        verbose(2, self.msg, lf=False)
        verbose(2, " continue ", lf=False)
        verbose(2, "end")
        verbose(0, "new %s" % self.msg)
        self.failUnlessEqual(self.sout.getvalue(),
                             "  %s continue end\nnew %s\n" % \
                             (self.msg, self.msg))

    def testCR(self):
        """Test if works fine with carriage return (cr) symbol"""
        verbose(2, self.msg, cr=True)
        verbose(2, "rewrite", cr=True)
        verbose(1, "rewrite 2", cr=True)
        verbose(1, " add", cr=False, lf=False)
        verbose(1, " finish")
        self.failUnlessEqual(self.sout.getvalue(),
                             '  %s\r              \rrewrite' % self.msg +\
                             '\r       \rrewrite 2 add finish\n')


    if __debug__:
        def testDebug(self):
            debug.active = [1, 2, 'SLC']
            debug('SLC', self.msg, lf=False)
            self.failUnlessEqual(self.sout.getvalue(),
                                 "[SLC] DEBUG: %s" % self.msg)

        # TODO: More tests needed for debug output testing

def suite():
    return unittest.makeSuite(VerboseOutputTest)


if __name__ == '__main__':
    import test_runner
