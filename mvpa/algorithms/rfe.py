#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Recursive feature elimination."""

__docformat__ = 'restructuredtext'

from mvpa.algorithms.featsel import FeatureSelection, \
                                    StopNBackHistoryCriterion, \
                                    FractionTailSelector
from numpy import arange

if __debug__:
    from mvpa.misc import debug

# TODO: Abs value of sensitivity should be able to rule RFE
# Often it is what abs value of the sensitivity is what matters.
# So we should either provide a simple decorator around arbitrary
# FeatureSelector to convert sensitivities to abs values before calling
# actual selector, or a decorator around SensitivityEstimators

class RFE(FeatureSelection):
    """ Recursive feature elimination.

    A `SensitivityAnalyzer` is used to compute sensitivity maps given a certain
    dataset. These sensitivity maps are in turn used to discard unimportant
    features. For each feature selection the transfer error on some testdatset
    is computed. This procedure is repeated until a given `StoppingCriterion` is
    reached.
    """

    # TODO: remove
    # doesn't work nicely -- if FeatureSelection defines its states via
    #                        _register_states, they would simply be ignored
    #_register_states = {'errors':True,
    #                    'nfeatures':True,
    #                    'history':True}

    def __init__(self,
                 sensitivity_analyzer,
                 transfer_error,
                 feature_selector=FractionTailSelector(0.05),
                 stopping_criterion=StopNBackHistoryCriterion(),
                 train_clf=True,
                 **kargs
                 ):
        # XXX Allow for multiple stopping criterions, e.g. error not decreasing
        # anymore OR number of features less than threshold
        """ Initialize recursive feature elimination

        Parameters
        ----------

        - `sensitivity_analyzer`: `SensitivityAnalyzer`
        - `transfer_error`: `TransferError` instance used to compute the
                transfer error of a classifier based on a certain feature set
                on test dataset.
        - `feature_selector`: Functor. Given a sensitivity map it has to return
                the ids of those features that should be kept.
        - `stopping_criterion`: Functor. Given a list of error values it has
                to return a tuple of two booleans. First values must indicate
                whether the criterion is fulfilled and the second value signals
                whether the latest error values is the total minimum.
        - `train_clf`: Flag whether the classifier in `transfer_error` should
                be trained before computing the error. In general this is
                required, but if the `sensitivity_analyzer` and
                `transfer_error` share and make use of the same classifier and
                can be switched off to save CPU cycles.
        """

        # base init first
        FeatureSelection.__init__(self, **kargs)

        self.__sensitivity_analyzer = sensitivity_analyzer
        """Sensitivity analyzer used to call at each step."""

        self.__transfer_error = transfer_error
        """Compute transfer error for each feature set."""

        self.__feature_selector = feature_selector
        """Functor which takes care about removing some features."""

        self.__stopping_criterion = stopping_criterion

        self.__train_clf = train_clf
        """Flag whether training classifier is required."""

        # register the state members
        self._registerState("errors")
        self._registerState("nfeatures")
        self._registerState("history")
        self._registerState("sensitivities", enabled=False)


    def __call__(self, dataset, testdataset, callables=[]):
        """Proceed and select the features recursively eliminating less
        important ones.

        Parameters
        ----------
        - `dataset`: `Dataset` used to compute sensitivity maps and train a
                classifier to determine the transfer error.
        - `testdataset`: `Dataset` used to test the trained classifer to
                determine the transfer error.

        Returns a new dataset with the feature subset of `dataset` that had the
        lowest transfer error of all tested sets until the stopping criterion
        was reached.
        """
        errors = []
        """Computed error for each tested features set."""

        if self.isStateEnabled("nfeatures"):
            self["nfeatures"] = []
            """Number of features at each step. Since it is not used by the
            algorithm it is stored directly in the state variable"""

        if self.isStateEnabled("history"):
            self["history"] = arange(dataset.nfeatures)
            """Store the last step # when the feature was still present
            """

        if self.isStateEnabled("sensitivities"):
            self["sensitivities"] = []

        stop = False
        """Flag when RFE should be stopped."""

        result = None
        """Will hold the best feature set ever."""

        wdataset = dataset
        """Operate on working dataset initially identical."""

        wtestdataset = testdataset
        """Same feature selection has to be performs on test dataset as well.
        This will hold the current testdataset."""

        step = 0
        """Counter how many selection step where done."""

        orig_feature_ids = arange(dataset.nfeatures)
        """List of feature Ids as per original dataset remaining at any given
        step"""

        while wdataset.nfeatures>0:
            if self.isStateEnabled("history"):
                # mark the features which are present at this step
                # if it brings anyb mentionable computational burden in the future,
                # only mark on removed features at each step
                self["history"][orig_feature_ids] = step

            # Compute sensitivity map
            # TODO add option to do RFE on a sensitivity map that is computed
            # a single time at the beginning of the process. This options
            # should then overwrite train_clf to always be True
            sensitivity = self.__sensitivity_analyzer(wdataset)

            if self.isStateEnabled("sensitivities"):
                self["sensitivities"].append(sensitivity)

            # do not retrain clf if not necessary
            if self.__train_clf:
                error = self.__transfer_error(wtestdataset, wdataset)
            else:
                error = self.__transfer_error(wtestdataset, None)

            # Record the error
            errors.append(error)

            # Check if it is time to stop and if we got
            # the best result
            (stop, isthebest) = self.__stopping_criterion(errors)

            nfeatures = wdataset.nfeatures

            if self.isStateEnabled("nfeatures"):
                self["nfeatures"].append(wdataset.nfeatures)

            if __debug__:
                debug('RFEC',
                      "Step %d: nfeatures=%d error=%.4f best/stop=%d/%d" %
                      (step, nfeatures, error, isthebest, stop))

            # store result
            if isthebest:
                result = wdataset
            # stop if it is time to finish
            if nfeatures == 1 or stop:
                break

            # Select features to preserve
            selected_ids = self.__feature_selector(sensitivity)
            # Create a dataset only with selected features
            wdataset = wdataset.selectFeatures(selected_ids)

            # need to update the test dataset as well
            # XXX why should it ever become None?
            # yoh: because we can have __transfer_error computed
            #      using wdataset. See xia-generalization estimate
            #      in lightsvm. Or for god's sake leave-one-out
            #      on a wdataset
            if not testdataset is None:
                wtestdataset = wtestdataset.selectFeatures(selected_ids)

            # provide evil access to internals :)
            for callable_ in callables:
                callable_(locals())

            step += 1

            # WARNING: THIS MUST BE THE LAST THING TO DO ON selected_ids
            selected_ids.sort()
            if self.isStateEnabled("history"):
                orig_feature_ids = orig_feature_ids[selected_ids]


        # charge state variable
        if self.isStateEnabled("errors"):
            self["errors"] = errors

        # best dataset ever is returned
        return result
