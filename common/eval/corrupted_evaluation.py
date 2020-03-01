import numpy
import sklearn.metrics
import common.utils
import math
import numpy
from .distal_evaluation import *


class CorruptedEvaluation(DistalEvaluation):
    """
    Out-of-distribution evaluation.
    """

    def __init__(self, clean_probabilities, corrupted_probabilities, labels, validation=0.1, include_misclassifications=False, detector=common.numpy.max_detector, clean_scores=None, corrupted_scores=None):
        """
        Constructor.

        For more details for confidence-thresholded evaluation, see adversarial_evaluation.py.

        :param clean_probabilities: probabilities on clean examles
        :type clean_probabilities: numpy.ndarray
        :param corrupted_probabilities: probabilities on adversarial examples
        :type corrupted_probabilities: numpy.ndarray
        :param validation: fraction of validation examples
        :type validation: float
        :param labels: labels
        :type labels: numpy.ndarray
        :param include_misclassifications: include mis classifications in confidence threshold computation
        :type include_misclassifications: bool
        :param detector: detector to apply on probabilities (default is taking the maximum confidence)
        :type detector: callable
        :param clean_scores: scores corresponding to clean_probabilities
        :type clean_scores: numpy.ndarray
        :param corrupted_scores: scores corresponding to clean_probabilities
        :type corrupted_scores: numpy.ndarray
        """

        assert validation >= 0
        labels = numpy.squeeze(labels)
        assert len(labels.shape) == 1

        assert len(clean_probabilities.shape) == 2
        assert clean_probabilities.shape[0] == labels.shape[0]
        assert clean_probabilities.shape[1] == numpy.max(labels) + 1
        assert len(corrupted_probabilities.shape) == len(clean_probabilities.shape) + 1
        assert corrupted_probabilities.shape[2] == clean_probabilities.shape[1]
        assert corrupted_probabilities.shape[1] <= clean_probabilities.shape[0]
        if validation > 0:
            assert corrupted_probabilities.shape[1] + int(validation * clean_probabilities.shape[0]) <= clean_probabilities.shape[0], (corrupted_probabilities.shape, clean_probabilities.shape, corrupted_probabilities.shape[1] + int(validation * clean_probabilities.shape[0]))
        assert not (detector is not None and (clean_scores is not None or corrupted_scores is not None))
        assert (corrupted_scores is not None and clean_scores is not None) or (corrupted_scores is None and clean_scores is None)

        self.reference_A = corrupted_probabilities.shape[0]
        """ (int) Attempts. """

        self.reference_N = corrupted_probabilities.shape[1]
        """ (int) Samples. """

        corrupted_probabilities = corrupted_probabilities.reshape(self.reference_A * self.reference_N, -1)
        reference_labels = numpy.tile(labels[:self.reference_N], self.reference_A)
        reference_probabilities = numpy.tile(clean_probabilities[:self.reference_N], (self.reference_A, 1))
        if clean_scores is not None:
            reference_scores = numpy.tile(clean_scores[:self.reference_N], (self.reference_A))

        assert corrupted_probabilities.shape[0] == reference_probabilities.shape[0], corrupted_probabilities.shape
        assert corrupted_probabilities.shape[1] == reference_probabilities.shape[1], corrupted_probabilities.shape
        assert len(corrupted_probabilities.shape) == len(reference_probabilities.shape)
        assert reference_probabilities.shape[0] == reference_labels.shape[0]

        corrupted_marginals = numpy.sum(corrupted_probabilities, axis=1)
        assert numpy.allclose(corrupted_marginals, numpy.ones(corrupted_marginals.shape))

        clean_marginals = numpy.sum(clean_probabilities, axis=1)
        assert numpy.allclose(clean_marginals, numpy.ones(clean_marginals.shape))

        corrupted_marginals = numpy.sum(corrupted_probabilities, axis=1)
        assert numpy.allclose(corrupted_marginals, numpy.ones(corrupted_marginals.shape)), corrupted_marginals

        self.reference_AN = self.reference_A * self.reference_N
        """ (int) Test examples. """

        assert reference_probabilities.shape[0] == self.reference_AN

        self.test_N = int((1 - validation) * clean_probabilities.shape[0])
        """ (int) Validation examples. """

        self.reference_probabilities = reference_probabilities
        """ (numpy.ndarray) Test probabilities. """

        self.reference_labels = reference_labels
        """ (numpy.ndarray) Test labels. """

        self.reference_predictions = numpy.argmax(self.reference_probabilities, axis=1)
        """ (numpy.ndarray) Test predicted labels. """

        self.reference_errors = (self.reference_predictions != self.reference_labels)
        """ (numpy.ndarray) Test errors. """

        # self.reference_confidences = self.reference_probabilities[numpy.arange(self.reference_predictions.shape[0]), self.reference_predictions]
        self.reference_confidences = None
        """ (numpy.ndarray) Test confidences. """

        if detector is not None:
            self.reference_confidences = detector(self.reference_probabilities)
        if clean_scores is not None:
            self.reference_confidences = reference_scores
            assert self.reference_confidences.shape[0] == self.reference_errors.shape[0], (self.reference_confidences.shape, self.reference_errors.shape)

        #
        self.test_probabilities = clean_probabilities[:self.test_N]
        """ (numpy.ndarray) Test probabilities. """

        self.test_labels = labels[:self.test_N]
        """ (numpy.ndarray) Test labels. """

        self.test_predictions = numpy.argmax(self.test_probabilities, axis=1)
        """ (numpy.ndarray) Test predicted labels. """

        self.test_errors = (self.test_predictions != self.test_labels)
        """ (numpy.ndarray) Test errors. """

        # self.test_confidences = self.test_probabilities[numpy.arange(self.test_predictions.shape[0]), self.test_predictions]
        self.test_confidences = None
        """ (numpy.ndarray) Test confidences. """

        if detector is not None:
            self.test_confidences = detector(self.test_probabilities)
        if clean_scores is not None:
            self.test_confidences = clean_scores[:self.test_N]
            assert self.test_confidences.shape[0] == self.test_errors.shape[0]

        self.validation_probabilities = None
        self.validation_confidences = None
        self.validation_predictions = None
        self.validation_errors = None
        self.validation_confidences = None

        if validation > 0:
            self.validation_probabilities = clean_probabilities[self.test_N:]
            """ (numpy.ndarray) Validation probabilities. """

            self.validation_labels = labels[self.test_N:]
            """ (numpy.ndarray) Validation labels."""

            self.validation_predictions = numpy.argmax(self.validation_probabilities, axis=1)
            """ (numpy.ndarray) Validation predicted labels. """

            self.validation_errors = (self.validation_predictions != self.validation_labels)
            """ (numpy.ndarray) Validation errors. """

            # self.validation_confidences = self.validation_probabilities[numpy.arange(self.validation_predictions.shape[0]), self.validation_predictions]
            self.validation_confidences = None
            """ (numpy.ndarray) Validation confidences. """

            if detector is not None:
                self.validation_confidences = detector(self.validation_probabilities)
            if clean_scores is not None:
                self.validation_confidences = clean_scores[self.test_N:]
                assert self.validation_confidences.shape[0] == self.validation_errors.shape[0]

        self.test_corrupted_probabilities = corrupted_probabilities
        """ (numpy.ndarray) Test probabilities. """

        self.test_corrupted_predictions = numpy.argmax(self.test_corrupted_probabilities, axis=1)
        """ (numpy.ndarray) Test predicted labels. """

        self.test_corrupted_errors = (self.test_corrupted_predictions != self.reference_labels)
        """ (numpy.ndarray) Test errors. """

        self.test_corrupted_confidences = None
        """ (numpy.ndarray) Test confidences. """

        if detector is not None:
            self.test_corrupted_confidences = detector(self.test_corrupted_probabilities)
        if clean_scores is not None:
            self.test_corrupted_confidences = corrupted_scores
            assert self.test_corrupted_confidences.shape[0] == self.test_corrupted_errors.shape[0], (self.test_corrupted_confidences.shape, self.test_corrupted_errors.shape)

        # NOTE: we do not need validation part of corrupted examples as threshold only depends on clean examples!

        # some cache variables
        self.sorted_validation_confidences = None
        """ (numpy.ndarray) Caches confidences. """

        self.include_misclassifications = include_misclassifications
        """ (bool) Include misclassifications. """

    def confidence_at_tpr(self, tpr):
        """
        Confidence threshold for given true positive rate.

        :param tpr: true positive rate in [0, 1]
        ;type tpr: float
        :return: confidence threshold
        :rtype: float
        """

        assert self.validation_confidences is not None
        assert tpr > 0

        # true positives are real examples
        if self.sorted_validation_confidences is None:
            if self.include_misclassifications:
                self.sorted_validation_confidences = numpy.sort(numpy.copy(self.validation_confidences))
            # This is computing the threshold only on correctly classified examples
            else:
                self.sorted_validation_confidences = numpy.sort(numpy.copy(self.validation_confidences[numpy.logical_not(self.validation_errors)]))
        # rounding is a hack see tests
        cutoff = math.floor(self.sorted_validation_confidences.shape[0] * round((1 - tpr), 2))
        assert cutoff >= 0
        assert cutoff < self.sorted_validation_confidences.shape[0]
        return self.sorted_validation_confidences[cutoff]

    def tpr_at_confidence(self, threshold):
        """
        True positive rate at confidence threshold.

        :param threshold: confidence threshold in [0, 1]
        :type threshold: float
        :return: false positive rate
        :rtype: float
        """

        return numpy.sum(self.test_confidences >= threshold) / float(self.test_N)

    def tnr_at_confidence(self, threshold):
        """
        True negative rate at confidence threshold.

        :param threshold: confidence threshold in [0, 1]
        :type threshold: float
        :return: false positive rate
        :rtype: float
        """

        return numpy.sum(self.test_corrupted_confidences <threshold) / float(self.reference_AN)

    def validation_tpr_at_confidence(self, threshold):
        """
        True positive rate at confidence threshold.

        :param threshold: confidence threshold in [0, 1]
        :type threshold: float
        :return: true positive rate
        :rtype: float
        """

        if self.include_misclassifications:
            validation_confidences = self.validation_confidences
        else:
            validation_confidences = self.validation_confidences[numpy.logical_not(self.validation_errors)]

        return numpy.sum(validation_confidences >= threshold) / float(validation_confidences.shape[0])

    def test_error(self):
        """
        Test error.

        :return: test error
        :rtype: float
        """

        return numpy.sum(self.test_errors.astype(int)) / float(self.test_N)

    def test_error_at_confidence(self, threshold):
        """
        Test error for given confidence threshold.

        :param threshold: confidence threshold
        :type threshold: float
        :return test error
        :rtype: float
        """

        nominator = numpy.sum(numpy.logical_and(self.test_errors, self.test_confidences >= threshold))
        denominator = numpy.sum(self.test_confidences >= threshold)
        if denominator > 0:
            return nominator / float(denominator)
        else:
            return 0

    def test_error_curve(self):
        """
        Test error for different confidence threshold.

        :return: test errors and confidences
        :rtype: numpy.ndarray, numpy.ndarray
        """

        scores = self.test_confidences
        sort = numpy.argsort(scores, axis=0)
        sorted_scores = scores[sort]

        test_errors = numpy.zeros((scores.shape[0]))
        thresholds = numpy.zeros((scores.shape[0]))

        for i in range(sort.shape[0]):
            thresholds[i] = sorted_scores[i]
            test_errors[i] = numpy.sum(self.test_errors[self.test_confidences >= thresholds[i]]) / float(numpy.sum(self.test_confidences >= thresholds[i]))

        return test_errors, thresholds

    def corrupted_test_error(self):
        """
        Test error.

        :return: test error
        :rtype: float
        """

        return numpy.sum(self.test_corrupted_errors.astype(int)) / float(self.reference_AN)

    def corrupted_test_error_at_confidence(self, threshold):
        """
        Test error for given confidence threshold.

        :param threshold: confidence threshold
        :type threshold: float
        :return test error
        :rtype: float
        """

        nominator = numpy.sum(numpy.logical_and(self.test_corrupted_errors, self.test_corrupted_confidences >= threshold))
        denominator = numpy.sum(self.test_corrupted_confidences >= threshold)
        if denominator > 0:
            return nominator / float(denominator)
        else:
            return 0

    def corrupted_test_error_curve(self):
        """
        Test error for different confidence threshold.

        :return: test errors and confidences
        :rtype: numpy.ndarray, numpy.ndarray
        """

        scores = self.test_corrupted_confidences
        sort = numpy.argsort(scores, axis=0)
        sorted_scores = scores[sort]

        test_errors = numpy.zeros((scores.shape[0]))
        thresholds = numpy.zeros((scores.shape[0]))

        for i in range(sort.shape[0]):
            thresholds[i] = sorted_scores[i]
            test_errors[i] = numpy.sum(self.test_corrupted_errors[self.test_corrupted_confidences >= thresholds[i]]) / float(numpy.sum(self.test_corrupted_confidences >= thresholds[i]))

        return test_errors, thresholds

    def fpr_at_confidence(self, threshold):
        """
        False positive rate at confidence threshold.

        :param threshold: confidence threshold in [0, 1]
        :type threshold: float
        :return: false positive rate
        :rtype: float
        """

        return numpy.sum(self.test_corrupted_confidences >= threshold) / float(self.reference_AN)

    def receiver_operating_characteristic_labels_scores(self):
        """
        Labels and scores for ROC.

        :return: labels and scores for sklearn.metrics.roc_auc_score
        :rtype: numpy.ndarray, numpy.ndarray
        """

        labels = numpy.concatenate((
            numpy.zeros(self.reference_AN),
            numpy.ones(self.test_N),
        ))
        scores = numpy.concatenate((
            self.test_corrupted_confidences,
            self.test_confidences
        ))

        return labels, scores

    def receiver_operating_characteristic_auc(self):
        """
        Computes the ROC curve for correct classified vs. incorrect classified.

        :return: ROC AUC score
        :rtype: float
        """

        labels, scores = self.receiver_operating_characteristic_labels_scores()
        if numpy.unique(labels).shape[0] == 1:
            return 1
        else:
            return sklearn.metrics.roc_auc_score(labels, scores)

    def receiver_operating_characteristic_curve(self):
        """
        Computes the ROC curve for correct classified vs. incorrect classified.

        :return: false positive rates, true positive rates, thresholds
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        labels, scores = self.receiver_operating_characteristic_labels_scores()
        return sklearn.metrics.roc_curve(labels, scores)

    def confidence_at_95tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.confidence_at_tpr(0.95)

    def confidence_at_98tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.confidence_at_tpr(0.98)

    def confidence_at_99tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.confidence_at_tpr(0.99)

    def confidence_at_995tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.confidence_at_tpr(0.995)

    def tpr_at_95tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tpr_at_confidence(self.confidence_at_tpr(0.95))

    def tpr_at_98tpr(self):
        """
        TPR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tpr_at_confidence(self.confidence_at_tpr(0.98))

    def tpr_at_99tpr(self):
        """
        TPR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tpr_at_confidence(self.confidence_at_tpr(0.99))

    def tpr_at_995tpr(self):
        """
        TPR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tpr_at_confidence(self.confidence_at_tpr(0.995))

    def tnr_at_95tpr(self):
        """
        TNR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tnr_at_confidence(self.confidence_at_tpr(0.95))

    def tnr_at_98tpr(self):
        """
        TNR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tnr_at_confidence(self.confidence_at_tpr(0.98))

    def tnr_at_99tpr(self):
        """
        TNR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tnr_at_confidence(self.confidence_at_tpr(0.99))

    def tnr_at_995tpr(self):
        """
        TNR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tnr_at_confidence(self.confidence_at_tpr(0.995))

    def validation_tpr_at_95tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.validation_tpr_at_confidence(self.confidence_at_tpr(0.95))

    def validation_tpr_at_98tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.validation_tpr_at_confidence(self.confidence_at_tpr(0.98))

    def validation_tpr_at_99tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.validation_tpr_at_confidence(self.confidence_at_tpr(0.99))

    def validation_tpr_at_995tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.validation_tpr_at_confidence(self.confidence_at_tpr(0.995))

    def fpr_at_95tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.fpr_at_confidence(self.confidence_at_tpr(0.95))

    def fpr_at_98tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.fpr_at_confidence(self.confidence_at_tpr(0.98))

    def fpr_at_99tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.fpr_at_confidence(self.confidence_at_tpr(0.99))

    def fpr_at_995tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.fpr_at_confidence(self.confidence_at_tpr(0.995))

    def test_error_at_95tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.test_error_at_confidence(self.confidence_at_tpr(0.95))

    def test_error_at_98tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.test_error_at_confidence(self.confidence_at_tpr(0.98))

    def test_error_at_99tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.test_error_at_confidence(self.confidence_at_tpr(0.99))

    def test_error_at_995tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.test_error_at_confidence(self.confidence_at_tpr(0.995))

    def corrupted_test_error_at_95tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.corrupted_test_error_at_confidence(self.confidence_at_tpr(0.95))

    def corrupted_test_error_at_98tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.corrupted_test_error_at_confidence(self.confidence_at_tpr(0.98))

    def corrupted_test_error_at_99tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.corrupted_test_error_at_confidence(self.confidence_at_tpr(0.99))

    def corrupted_test_error_at_995tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.corrupted_test_error_at_confidence(self.confidence_at_tpr(0.995))