#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SplittingSensitivityAnalyzer"""

import unittest

import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.algorithms.featsel import FixedNElementTailSelector, FeatureSelectionPipeline, FractionTailSelector
from mvpa.algorithms.linsvmweights import LinearSVMWeights
from mvpa.clfs import *
from mvpa.clfs.svm import LinearNuSVMC
from mvpa.datasets.splitter import NFoldSplitter
from mvpa.algorithms.datameasure import *

from tests_warehouse import *

class SensitivityAnalysersTests(unittest.TestCase):

    def setUp(self):
        self.nonbogus = [1, 3]          # informative features
        self.nfeatures = 5              # total features
        self.dataset = normalFeatureDataset(perlabel=200, nlabels=2,
                                            nfeatures=self.nfeatures,
                                            nonbogus_features=self.nonbogus,
                                            snr=6)


    def testAnalyzerWithSplitClassifier(self):
        svm = LinearNuSVMC()

        #svm_weigths = LinearSVMWeights(svm)

        # assumming many defaults it is as simple as
        sana = selectAnalyzer( SplitClassifier(clf=svm),
                               enable_states=["sensitivities"] ) # and lets look at all sensitivities

        # and we get sensitivity analyzer which works on splits and uses
        # linear svm sensitivity
        map_ = sana(self.dataset)
        self.failUnless(len(map_) == self.dataset.nfeatures)

        for conf_matrix in [sana.clf.trained_confusion] + sana.clf.trained_confusions.matrices:
            self.failUnless(conf_matrix.percentCorrect>85,
                            msg="We must have trained on each one more or less correctly")

        errors = [x.percentCorrect for x in sana.clf.trained_confusions.matrices]

        self.failUnless(N.min(errors) != N.max(errors),
                        msg="Splits should have slightly but different generalization")

        # lets go through all sensitivities and see if we selected the right features
        for map__ in [map_] + sana.combined_analyzer.sensitivities:
            self.failUnlessEqual(
                list(FixedNElementTailSelector(self.nfeatures - len(self.nonbogus))(map__)),
                list(self.nonbogus),
                msg="At the end we should have selected the right features")


    def __testFSPipelineWithAnalyzerWithSplitClassifier(self):
        basic_clf = LinearNuSVMC()
        multi_clf = MulticlassClassifier(clf=basic_clf)
        #svm_weigths = LinearSVMWeights(svm)

        # Proper RFE: aggregate sensitivities across multiple splits,
        # but also due to multi class those need to be aggregated
        # somehow. Transfer error here should be 'leave-1-out' error
        # of split classifier itself
        rfe = RFE(sensitivity_analyzer=Absolute(
                      selectAnalyzer(SplitClassifier(clf=svm),
                                     enable_states=["sensitivities"])),
                  transfer_error=trans_error,
                  feature_selector=FeatureSelectionPipeline(
                      [FractionTailSelector(0.5),
                       FixedNElementTailSelector(1)]),
                  train_clf=True)

        # assumming many defaults it is as simple as
        sana = selectAnalyzer( SplitClassifier(clf=svm),
                               enable_states=["sensitivities"] ) # and lets look at all sensitivities

        # and we get sensitivity analyzer which works on splits and uses
        # linear svm sensitivity
        selected_features = rfe(self.dataset)


def suite():
    return unittest.makeSuite(SensitivityAnalysersTests)


if __name__ == '__main__':
    import test_runner
