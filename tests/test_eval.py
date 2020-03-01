import unittest
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import common.numpy
import common.eval
import random


class TestEval(unittest.TestCase):
    def distributionAt(self, label, confidence, labels=10):
        probabilities = [0] * labels
        probabilities[label] = confidence
        for i in range(len(probabilities)):
            if i == label:
                continue
            probabilities[i] = (1 - confidence) / (labels - 1)
        self.assertAlmostEqual(1, numpy.sum(probabilities))
        return probabilities

    def testCleanEvaluationNotCorrectShape(self):
        labels = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        probabilities = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertRaises(AssertionError, common.eval.CleanEvaluation, probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertRaises(AssertionError, common.eval.CleanEvaluation, probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.assertRaises(AssertionError, common.eval.CleanEvaluation, probabilities, labels)

    def testCleanEvaluationNotCorrectClasses(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.assertRaises(AssertionError, common.eval.CleanEvaluation, probabilities, labels)

    def testCleanEvaluationNotProbabilities(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09],
            [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.assertRaises(AssertionError, common.eval.CleanEvaluation, probabilities, labels)

    def testCleanEvaluationTestErrorNoValidation(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # ok
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # ok
            [0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], # slightly different
            [0.099, 0.099, 0.099, 0.109, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099], # hard
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        eval = common.eval.CleanEvaluation(probabilities, labels, validation=0)
        self.assertEqual(eval.test_N, eval.N)
        self.assertEqual(eval.N, 10)
        self.assertEqual(eval.test_error(), 0)

        test_errors = [
            1,
            7,
            99,
            73,
        ]

        for test_error in test_errors:
            labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            labels = numpy.tile(labels, 10)
            probabilities = common.numpy.one_hot(labels, 10)

            indices = numpy.array(random.sample(range(100), test_error))
            self.assertEqual(numpy.unique(indices).shape[0], indices.shape[0])

            for i in indices:
                probabilities[i] = numpy.flip(probabilities[i])

            eval = common.eval.CleanEvaluation(probabilities, labels, validation=0)
            self.assertEqual(eval.test_error(), test_error/100)
            self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

            for threshold in numpy.linspace(0, 1, 50):
                self.assertEqual(eval.test_error_at_confidence(threshold), test_error/100)

    def testCleanEvaluationTestErrorValidation(self):
        test_errors = [
            1,
            7,
            89,
            73,
        ]

        for test_error in test_errors:
            labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            labels = numpy.tile(labels, 10)
            self.assertTrue(labels.shape[0], 110)
            probabilities = common.numpy.one_hot(labels, 10)

            indices = numpy.array(random.sample(range(90), test_error))
            self.assertEqual(numpy.unique(indices).shape[0], indices.shape[0])

            for i in indices:
                probabilities[i] = numpy.flip(probabilities[i])

            eval = common.eval.CleanEvaluation(probabilities, labels, validation=0.1)
            self.assertEqual(eval.N, 100)
            self.assertEqual(eval.test_N, 90)
            self.assertEqual(eval.validation_N, 10)
            self.assertAlmostEqual(eval.test_error(), test_error/90)

            for threshold in numpy.linspace(0, 1, 50):
                self.assertAlmostEqual(eval.test_error_at_confidence(threshold), test_error/90)

    def testCleanEvaluationTestErrorAtConfidence(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        labels = numpy.tile(labels, 10)
        probabilities = common.numpy.one_hot(labels, 10)

        i = 0
        for probability in numpy.linspace(0.1, 1, 101)[1:]:
            probabilities_i = probabilities[i]
            probabilities_i *= probability
            probabilities_i[probabilities_i == 0] = (1 - probability)/9
            probabilities[i] = probabilities_i
            i += 1

        eval = common.eval.CleanEvaluation(probabilities, labels, validation=0)
        self.assertEqual(eval.test_error(), 0)

        for threshold in numpy.linspace(0, 1, 100):
            self.assertAlmostEqual(eval.test_error_at_confidence(threshold), 0)

        test_errors = [
            1,
            7,
            73,
            99,
        ]

        for test_error in test_errors:
            labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            labels = numpy.tile(labels, 10)
            probabilities = common.numpy.one_hot(labels, 10)

            i = 0
            self.assertEqual(numpy.linspace(0.11, 1, 101)[1:].shape[0], probabilities.shape[0])
            for probability in numpy.linspace(0.11, 1, 101)[1:]:
                probabilities_i = probabilities[i]
                label = numpy.argmax(probabilities_i)
                probabilities_i *= probability
                probabilities_i[probabilities_i == 0] = (1 - probability) / 9
                probabilities[i] = probabilities_i
                self.assertEqual(label, labels[i])
                self.assertEqual(labels[i], numpy.argmax(probabilities[i]))
                i += 1

            numpy.testing.assert_array_equal(labels, numpy.argmax(probabilities, axis=1))

            indices = numpy.array(random.sample(range(100), test_error))
            self.assertEqual(numpy.unique(indices).shape[0], indices.shape[0])

            for i in indices:
                probabilities_i = probabilities[i]
                label = numpy.argmax(probabilities_i)
                probabilities_i = numpy.zeros(10)
                probabilities_i[label] = 1
                probabilities_i = numpy.flip(probabilities_i)
                probabilities[i] = probabilities_i
                self.assertEqual(label, labels[i])
                self.assertNotEqual(labels[i], numpy.argmax(probabilities[i]))

            eval = common.eval.CleanEvaluation(probabilities, labels, validation=0)
            self.assertEqual(eval.N, 100)
            self.assertEqual(eval.test_N, 100)
            self.assertEqual(eval.validation_N, 0)
            self.assertEqual(eval.test_error(), test_error/100)

            for threshold in numpy.linspace(0, 1, 100):
                self.assertAlmostEqual(eval.test_error_at_confidence(threshold), test_error/numpy.sum(numpy.max(probabilities, axis=1) >= threshold))

    def testCleanEvaluationTPRAndFPR(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(9, 1),
            self.distributionAt(9, 1),
            self.distributionAt(9, 1),
            self.distributionAt(9, 1),
            self.distributionAt(9, 1),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
            self.distributionAt(9, 1),
            self.distributionAt(9, 1),
            self.distributionAt(9, 1),
            self.distributionAt(9, 1),
            self.distributionAt(9, 1),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
        ])
        eval = common.eval.CleanEvaluation(probabilities, labels, validation=0.5)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.2), 0.9)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.4), 0.8)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.6), 0.7)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.8), 0.6)
        self.assertAlmostEqual(eval.confidence_at_tpr(1), 0.5)

        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

        for threshold in numpy.linspace(0, 0.499, 100):
            self.assertAlmostEqual(eval.tpr_at_confidence(threshold), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(0.6), 4. / 5.)
        self.assertAlmostEqual(eval.tpr_at_confidence(0.7), 3. / 5.)
        self.assertAlmostEqual(eval.tpr_at_confidence(0.8), 2. / 5.)
        self.assertAlmostEqual(eval.tpr_at_confidence(0.9), 1. / 5.)
        self.assertAlmostEqual(eval.tpr_at_confidence(0.91), 0. / 5.)

        for threshold in numpy.linspace(0, 1, 100):
            self.assertAlmostEqual(eval.fpr_at_confidence(threshold), 1)

    def testAdversarialEvaluationNotCorrectShape(self):
        labels = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        probabilities = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.AdversarialEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        adversarial_probabilities = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.AdversarialEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.AdversarialEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        adversarial_probabilities = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.AdversarialEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.AdversarialEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.assertRaises(AssertionError, common.eval.AdversarialEvaluation, probabilities, adversarial_probabilities, labels)

    def testAdversarialEvaluationCorrectShapes(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ])
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.reference_labels.shape[0], 3*10)
        self.assertEqual(eval.reference_probabilities.shape[0], 3 * 10)
        self.assertEqual(eval.reference_confidences.shape[0], 3 * 10)
        self.assertEqual(eval.reference_errors.shape[0], 3 * 10)
        self.assertEqual(eval.test_adversarial_probabilities.shape[0], 3 * 10)
        self.assertEqual(eval.test_adversarial_confidences.shape[0], 3 * 10)
        self.assertEqual(eval.test_adversarial_errors.shape[0], 3 * 10)
        numpy.testing.assert_array_equal(eval.reference_predictions, numpy.tile(labels, 3))
        self.assertEqual(numpy.sum(eval.reference_errors), 0)
        self.assertEqual(numpy.sum(eval.test_adversarial_errors), 30 - 3)

    def testAdversarialEvaluationCorrectShapesWithErrors(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ])
        errors = numpy.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0, errors=errors)
        self.assertEqual(eval.test_labels.shape[0], 10)
        self.assertEqual(eval.test_probabilities.shape[0], 10)
        self.assertEqual(eval.test_confidences.shape[0], 10)
        self.assertEqual(eval.test_errors.shape[0], 10)
        self.assertEqual(eval.test_adversarial_probabilities.shape[0], 10)
        self.assertEqual(eval.test_adversarial_confidences.shape[0], 10)
        self.assertEqual(eval.test_adversarial_errors.shape[0], 10)
        self.assertEqual(numpy.sum(eval.test_errors), 0)
        self.assertEqual(numpy.sum(eval.test_adversarial_errors), 10 - 1)

        errors = numpy.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0, errors=errors)
        self.assertEqual(eval.test_labels.shape[0], 10)
        self.assertEqual(eval.test_probabilities.shape[0], 10)
        self.assertEqual(eval.test_confidences.shape[0], 10)
        self.assertEqual(eval.test_errors.shape[0], 10)
        self.assertEqual(eval.test_adversarial_probabilities.shape[0], 10)
        self.assertEqual(eval.test_adversarial_confidences.shape[0], 10)
        self.assertEqual(eval.test_adversarial_errors.shape[0], 10)
        self.assertEqual(numpy.sum(eval.test_errors), 0)
        self.assertEqual(numpy.sum(eval.test_adversarial_errors), 10 - 1)

    def testAdversarialEvaluationNotCorrectClasses(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.assertRaises(AssertionError, common.eval.AdversarialEvaluation, probabilities, adversarial_probabilities, labels)

    def testAdversarialEvaluationNotProbabilities(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09],
            [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.AdversarialEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09],
            [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.AdversarialEvaluation, probabilities, adversarial_probabilities, labels)

    def testAdversarialEvaluationTestErrorNoValidation(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # ok
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # ok
            [0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], # slightly different
            [0.099, 0.099, 0.099, 0.109, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099], # hard
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ok
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # ok
            [0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # slightly different
            [0.099, 0.099, 0.099, 0.109, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099],  # hard
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.test_N, probabilities.shape[0])
        self.assertEqual(eval.reference_AN, 10)
        self.assertEqual(eval.reference_A, 1)
        self.assertEqual(eval.reference_N, 10)
        self.assertEqual(eval.test_error(), 0)

        test_errors = [
            1,
            7,
            99,
            73,
        ]

        for test_error in test_errors:
            labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            labels = numpy.tile(labels, 10)
            probabilities = common.numpy.one_hot(labels, 10)
            adversarial_probabilities = numpy.array([common.numpy.one_hot(labels, 10)])

            indices = numpy.array(random.sample(range(100), test_error))
            self.assertEqual(numpy.unique(indices).shape[0], indices.shape[0])

            for i in indices:
                probabilities[i] = numpy.flip(probabilities[i])

            eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
            self.assertEqual(eval.test_error(), test_error/100)
            self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

            for threshold in numpy.linspace(0, 1, 50):
                self.assertEqual(eval.test_error_at_confidence(threshold), test_error/100)

    def testAdversarialEvaluationTestErrorNoValidationTrivialScores(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # ok
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # ok
            [0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], # slightly different
            [0.099, 0.099, 0.099, 0.109, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099], # hard
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ok
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # ok
            [0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # slightly different
            [0.099, 0.099, 0.099, 0.109, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099],  # hard
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        scores = numpy.max(probabilities, axis=1)
        adversarial_scores = numpy.max(adversarial_probabilities, axis=2)
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0, errors=None,
                                                 include_misclassifications=False, detector=None, clean_scores=scores,
                                                 adversarial_scores=adversarial_scores)
        self.assertEqual(eval.test_N, probabilities.shape[0])
        self.assertEqual(eval.reference_AN, 10)
        self.assertEqual(eval.reference_A, 1)
        self.assertEqual(eval.reference_N, 10)
        self.assertEqual(eval.test_error(), 0)

        test_errors = [
            1,
            7,
            99,
            73,
        ]

        for test_error in test_errors:
            labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            labels = numpy.tile(labels, 10)
            probabilities = common.numpy.one_hot(labels, 10)
            adversarial_probabilities = numpy.array([common.numpy.one_hot(labels, 10)])

            indices = numpy.array(random.sample(range(100), test_error))
            self.assertEqual(numpy.unique(indices).shape[0], indices.shape[0])

            for i in indices:
                probabilities[i] = numpy.flip(probabilities[i])

            scores = numpy.max(probabilities, axis=1)
            adversarial_scores = numpy.max(adversarial_probabilities, axis=2)
            eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0, errors=None,
                                                     include_misclassifications=False, detector=None, clean_scores=scores,
                                                     adversarial_scores=adversarial_scores)
            self.assertEqual(eval.test_error(), test_error/100)
            self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

            for threshold in numpy.linspace(0, 1, 50):
                self.assertEqual(eval.test_error_at_confidence(threshold), test_error/100)

    def testAdversarialEvaluationSuccessRateNoValidation(self):
        success_rates = [
            1,
            7,
            73,
        ]

        for success_rate in success_rates:
            labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            labels = numpy.tile(labels, 10)
            probabilities = common.numpy.one_hot(labels, 10)

            # add 2 test errors
            labels[-2] = 0
            labels[-1] = 0

            adversarial_probabilities = numpy.array([common.numpy.one_hot(labels, 10)])

            indices = numpy.array(random.sample(range(98), success_rate))
            self.assertEqual(numpy.unique(indices).shape[0], indices.shape[0])

            for i in indices:
                adversarial_probabilities[0][i] = numpy.flip(adversarial_probabilities[0][i])

            eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
            self.assertEqual(eval.test_error(), 2/100.)
            self.assertEqual(numpy.sum(eval.reference_errors), 2)
            self.assertEqual(eval.success_rate(), success_rate/(100. - 2.))
            # here all regular images are classified correctly!
            self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

            for threshold in numpy.linspace(0, 1, 50):
                self.assertEqual(eval.test_error_at_confidence(threshold), 2/100.)
                #self.assertEqual(eval.success_rate_at_confidence(threshold), success_rate/(100. - 2.))

    def testAdversarialEvaluationSuccessRateNoValidationTrivialScores(self):
        success_rates = [
            1,
            7,
            73,
        ]

        for success_rate in success_rates:
            labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            labels = numpy.tile(labels, 10)
            probabilities = common.numpy.one_hot(labels, 10)

            # add 2 test errors
            labels[-2] = 0
            labels[-1] = 0

            adversarial_probabilities = numpy.array([common.numpy.one_hot(labels, 10)])

            indices = numpy.array(random.sample(range(98), success_rate))
            self.assertEqual(numpy.unique(indices).shape[0], indices.shape[0])

            for i in indices:
                adversarial_probabilities[0][i] = numpy.flip(adversarial_probabilities[0][i])

            scores = numpy.max(probabilities, axis=1)
            adversarial_scores = numpy.max(adversarial_probabilities, axis=2)
            eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0, errors=None,
                                                     include_misclassifications=False, detector=None, clean_scores=scores,
                                                     adversarial_scores=adversarial_scores)

            self.assertEqual(eval.test_error(), 2/100.)
            self.assertEqual(numpy.sum(eval.reference_errors), 2)
            self.assertEqual(eval.success_rate(), success_rate/(100. - 2.))
            # here all regular images are classified correctly!
            self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

            for threshold in numpy.linspace(0, 1, 50):
                self.assertEqual(eval.test_error_at_confidence(threshold), 2/100.)
                #self.assertEqual(eval.success_rate_at_confidence(threshold), success_rate/(100. - 2.))

    def testAdversarialEvaluationRobustTestErrorNoValidation(self):
        robust_test_errors = [
            1,
            7,
            99,
            73,
        ]

        for robust_test_error in robust_test_errors:
            labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            labels = numpy.tile(labels, 10)
            probabilities = common.numpy.one_hot(labels, 10)
            adversarial_probabilities = numpy.array([common.numpy.one_hot(labels, 10)])

            indices = numpy.array(random.sample(range(100), robust_test_error))
            self.assertEqual(numpy.unique(indices).shape[0], indices.shape[0])

            for i in indices:
                adversarial_probabilities[0][i] = numpy.flip(adversarial_probabilities[0][i])

            eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
            self.assertEqual(eval.test_error(), 0)
            self.assertEqual(eval.robust_test_error(), robust_test_error/100.)
            self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

            for threshold in numpy.linspace(0, 1, 50):
                self.assertEqual(eval.test_error_at_confidence(threshold), 0)
                self.assertEqual(eval.robust_test_error_at_confidence(threshold), robust_test_error/100.)
                # only the same because we do not have the special case of adversarial examples getting higher
                # confidence than the corresponding test examples
                self.assertEqual(eval.alternative_robust_test_error_at_confidence(threshold), robust_test_error / 100.)

    def testAdversarialEvaluationRobustTestErrorNoValidationTrivialScores(self):
        robust_test_errors = [
            1,
            7,
            99,
            73,
        ]

        for robust_test_error in robust_test_errors:
            labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            labels = numpy.tile(labels, 10)
            probabilities = common.numpy.one_hot(labels, 10)
            adversarial_probabilities = numpy.array([common.numpy.one_hot(labels, 10)])

            indices = numpy.array(random.sample(range(100), robust_test_error))
            self.assertEqual(numpy.unique(indices).shape[0], indices.shape[0])

            for i in indices:
                adversarial_probabilities[0][i] = numpy.flip(adversarial_probabilities[0][i])

            scores = numpy.max(probabilities, axis=1)
            adversarial_scores = numpy.max(adversarial_probabilities, axis=2)
            eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0, errors=None,
                                                     include_misclassifications=False, detector=None, clean_scores=scores,
                                                     adversarial_scores=adversarial_scores)
            self.assertEqual(eval.test_error(), 0)
            self.assertEqual(eval.robust_test_error(), robust_test_error/100.)
            self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

            for threshold in numpy.linspace(0, 1, 50):
                self.assertEqual(eval.test_error_at_confidence(threshold), 0)
                self.assertEqual(eval.robust_test_error_at_confidence(threshold), robust_test_error/100.)

    def testAdversarialEvaluationRobustTestErrorAtNoValidation(self):
        # simple case where test example confidences always higher than adversarial example confidences
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(1, 0.4),
            self.distributionAt(2, 0.45),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.55),
            self.distributionAt(5, 0.6),
            self.distributionAt(6, 0.65),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.75),
            self.distributionAt(9, 0.8),
            self.distributionAt(0, 0.85),
        ]])
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.test_error(), 0)
        self.assertEqual(eval.robust_test_error(), 1)

        self.assertEqual(eval.test_error_at_confidence(0), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0), 1)
        self.assertEqual(eval.test_error_at_confidence(0.4), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.4), 1)
        self.assertEqual(eval.test_error_at_confidence(0.45), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.45), 0.9)
        self.assertEqual(eval.test_error_at_confidence(0.5), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.5), 0.8)
        self.assertEqual(eval.test_error_at_confidence(0.55), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.55), 0.7)
        self.assertEqual(eval.test_error_at_confidence(0.6), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.6), 0.6)
        self.assertEqual(eval.test_error_at_confidence(0.65), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.65), 0.5)
        self.assertEqual(eval.test_error_at_confidence(0.7), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.7), 0.4)
        self.assertEqual(eval.test_error_at_confidence(0.75), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.75), 0.3)
        self.assertEqual(eval.test_error_at_confidence(0.8), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.8), 0.2)
        self.assertEqual(eval.test_error_at_confidence(0.85), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.85), 0.1)
        self.assertEqual(eval.test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.test_error_at_confidence(0.95), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.95), 0)

        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

        # more complex case where with certain threshold half of the test examples "fall out"
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.7),
            self.distributionAt(1, 0.7),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.7),
            self.distributionAt(4, 0.7),
            self.distributionAt(5, 0.7),
            self.distributionAt(6, 0.7),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.7),
            self.distributionAt(9, 0.7),
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(1, 0.4),
            self.distributionAt(2, 0.45),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.55),
            self.distributionAt(5, 0.6),
            self.distributionAt(6, 0.65),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.75),
            self.distributionAt(9, 0.8),
            self.distributionAt(0, 0.85),
        ]])
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.test_error(), 0)
        self.assertEqual(eval.robust_test_error(), 1)

        self.assertEqual(eval.test_error_at_confidence(0), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0), 1)
        self.assertEqual(eval.test_error_at_confidence(0.4), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.4), 1)
        self.assertEqual(eval.test_error_at_confidence(0.45), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.45), 0.9)
        self.assertEqual(eval.test_error_at_confidence(0.5), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.5), 0.8)
        self.assertEqual(eval.test_error_at_confidence(0.55), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.55), 0.7)
        self.assertEqual(eval.test_error_at_confidence(0.6), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.6), 0.6)
        self.assertEqual(eval.test_error_at_confidence(0.65), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.65), 0.5)
        self.assertEqual(eval.test_error_at_confidence(0.7), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.7), 0.4)
        # cases where denominator is zero for test error
        self.assertEqual(eval.test_error_at_confidence(0.75), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.75), 1)
        self.assertEqual(eval.test_error_at_confidence(0.8), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.8), 1)
        self.assertEqual(eval.test_error_at_confidence(0.85), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.85), 1)
        # cases where denominator is zero for ROBUST test error
        self.assertEqual(eval.test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.test_error_at_confidence(0.95), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.95), 0)

        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

    def testAdversarialEvaluationRobustTestErrorAtNoValidationTrivialScores(self):
        # simple case where test example confidences always higher than adversarial example confidences
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(1, 0.4),
            self.distributionAt(2, 0.45),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.55),
            self.distributionAt(5, 0.6),
            self.distributionAt(6, 0.65),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.75),
            self.distributionAt(9, 0.8),
            self.distributionAt(0, 0.85),
        ]])
        scores = numpy.max(probabilities, axis=1)
        adversarial_scores = numpy.max(adversarial_probabilities, axis=2)
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0, errors=None,
                                                 include_misclassifications=False, detector=None, clean_scores=scores,
                                                 adversarial_scores=adversarial_scores)
        self.assertEqual(eval.test_error(), 0)
        self.assertEqual(eval.robust_test_error(), 1)

        self.assertEqual(eval.test_error_at_confidence(0), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0), 1)
        self.assertEqual(eval.test_error_at_confidence(0.4), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.4), 1)
        self.assertEqual(eval.test_error_at_confidence(0.45), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.45), 0.9)
        self.assertEqual(eval.test_error_at_confidence(0.5), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.5), 0.8)
        self.assertEqual(eval.test_error_at_confidence(0.55), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.55), 0.7)
        self.assertEqual(eval.test_error_at_confidence(0.6), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.6), 0.6)
        self.assertEqual(eval.test_error_at_confidence(0.65), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.65), 0.5)
        self.assertEqual(eval.test_error_at_confidence(0.7), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.7), 0.4)
        self.assertEqual(eval.test_error_at_confidence(0.75), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.75), 0.3)
        self.assertEqual(eval.test_error_at_confidence(0.8), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.8), 0.2)
        self.assertEqual(eval.test_error_at_confidence(0.85), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.85), 0.1)
        self.assertEqual(eval.test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.test_error_at_confidence(0.95), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.95), 0)

        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

        # more complex case where with certain threshold half of the test examples "fall out"
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.7),
            self.distributionAt(1, 0.7),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.7),
            self.distributionAt(4, 0.7),
            self.distributionAt(5, 0.7),
            self.distributionAt(6, 0.7),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.7),
            self.distributionAt(9, 0.7),
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(1, 0.4),
            self.distributionAt(2, 0.45),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.55),
            self.distributionAt(5, 0.6),
            self.distributionAt(6, 0.65),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.75),
            self.distributionAt(9, 0.8),
            self.distributionAt(0, 0.85),
        ]])
        scores = numpy.max(probabilities, axis=1)
        adversarial_scores = numpy.max(adversarial_probabilities, axis=2)
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0, errors=None,
                                                 include_misclassifications=False, detector=None, clean_scores=scores,
                                                 adversarial_scores=adversarial_scores)
        self.assertEqual(eval.test_error(), 0)
        self.assertEqual(eval.robust_test_error(), 1)

        self.assertEqual(eval.test_error_at_confidence(0), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0), 1)
        self.assertEqual(eval.test_error_at_confidence(0.4), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.4), 1)
        self.assertEqual(eval.test_error_at_confidence(0.45), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.45), 0.9)
        self.assertEqual(eval.test_error_at_confidence(0.5), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.5), 0.8)
        self.assertEqual(eval.test_error_at_confidence(0.55), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.55), 0.7)
        self.assertEqual(eval.test_error_at_confidence(0.6), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.6), 0.6)
        self.assertEqual(eval.test_error_at_confidence(0.65), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.65), 0.5)
        self.assertEqual(eval.test_error_at_confidence(0.7), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.7), 0.4)
        # cases where denominator is zero for test error
        self.assertEqual(eval.test_error_at_confidence(0.75), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.75), 1)
        self.assertEqual(eval.test_error_at_confidence(0.8), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.8), 1)
        self.assertEqual(eval.test_error_at_confidence(0.85), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.85), 1)
        # cases where denominator is zero for ROBUST test error
        self.assertEqual(eval.test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.test_error_at_confidence(0.95), 0)
        self.assertEqual(eval.robust_test_error_at_confidence(0.95), 0)

        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

    def testAdversarialEvaluationROCAUCNoValidation(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.7),
            self.distributionAt(1, 0.7),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.7),
            self.distributionAt(4, 0.7),
            self.distributionAt(5, 0.7),
            self.distributionAt(6, 0.7),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.7),
            self.distributionAt(9, 0.7),
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(1, 0.5),
            self.distributionAt(2, 0.5),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.5),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.5),
            self.distributionAt(7, 0.5),
            self.distributionAt(8, 0.5),
            self.distributionAt(9, 0.5),
            self.distributionAt(0, 0.5),
        ]])
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.test_error(), 0)
        self.assertEqual(eval.robust_test_error(), 1)
        self.assertEqual(eval.receiver_operating_characteristic_auc(), 1)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(1, 0.5),
            self.distributionAt(2, 0.5),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.5),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.5),
            self.distributionAt(7, 0.5),
            self.distributionAt(8, 0.5),
            self.distributionAt(9, 0.5),
            self.distributionAt(0, 0.5),

        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(0, 0.7),
            self.distributionAt(1, 0.7),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.7),
            self.distributionAt(4, 0.7),
            self.distributionAt(5, 0.7),
            self.distributionAt(6, 0.7),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.7),
            self.distributionAt(9, 0.7),
        ]])
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.test_error(), 1)
        self.assertEqual(eval.robust_test_error(), 1)
        self.assertEqual(eval.receiver_operating_characteristic_auc(), 1)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

    def testAdversarialEvaluationROCAUCNoValidationTrivialScores(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.7),
            self.distributionAt(1, 0.7),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.7),
            self.distributionAt(4, 0.7),
            self.distributionAt(5, 0.7),
            self.distributionAt(6, 0.7),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.7),
            self.distributionAt(9, 0.7),
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(1, 0.5),
            self.distributionAt(2, 0.5),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.5),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.5),
            self.distributionAt(7, 0.5),
            self.distributionAt(8, 0.5),
            self.distributionAt(9, 0.5),
            self.distributionAt(0, 0.5),
        ]])
        scores = numpy.max(probabilities, axis=1)
        adversarial_scores = numpy.max(adversarial_probabilities, axis=2)
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0, errors=None,
                                                 include_misclassifications=False, detector=None, clean_scores=scores,
                                                 adversarial_scores=adversarial_scores)
        self.assertEqual(eval.test_error(), 0)
        self.assertEqual(eval.robust_test_error(), 1)
        self.assertEqual(eval.receiver_operating_characteristic_auc(), 1)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(1, 0.5),
            self.distributionAt(2, 0.5),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.5),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.5),
            self.distributionAt(7, 0.5),
            self.distributionAt(8, 0.5),
            self.distributionAt(9, 0.5),
            self.distributionAt(0, 0.5),

        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(0, 0.7),
            self.distributionAt(1, 0.7),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.7),
            self.distributionAt(4, 0.7),
            self.distributionAt(5, 0.7),
            self.distributionAt(6, 0.7),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.7),
            self.distributionAt(9, 0.7),
        ]])
        scores = numpy.max(probabilities, axis=1)
        adversarial_scores = numpy.max(adversarial_probabilities, axis=2)
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0, errors=None,
                                                 include_misclassifications=False, detector=None, clean_scores=scores,
                                                 adversarial_scores=adversarial_scores)
        self.assertEqual(eval.test_error(), 1)
        self.assertEqual(eval.robust_test_error(), 1)
        self.assertEqual(eval.receiver_operating_characteristic_auc(), 1)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

    def testAdversarialEvaluationTPRAndFPR(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
        ]])
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0.5)
        self.assertAlmostEqual(eval.test_error(), 0)
        self.assertAlmostEqual(eval.robust_test_error(), 1)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.2), 0.9)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.4), 0.8)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.6), 0.7)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.8), 0.6)
        self.assertAlmostEqual(eval.confidence_at_tpr(1), 0.5)

        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

    def testAdversarialEvaluationTPRAndFPRTrivialScores(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
        ]])
        scores = numpy.max(probabilities, axis=1)
        adversarial_scores = numpy.max(adversarial_probabilities, axis=2)
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0.5, errors=None,
                                                 include_misclassifications=False, detector=None, clean_scores=scores,
                                                 adversarial_scores=adversarial_scores)
        self.assertAlmostEqual(eval.test_error(), 0)
        self.assertAlmostEqual(eval.robust_test_error(), 1)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.2), 0.9)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.4), 0.8)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.6), 0.7)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.8), 0.6)
        self.assertAlmostEqual(eval.confidence_at_tpr(1), 0.5)

        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

    def testAdversarialEvaluationTPRAndFPRScores(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
        ]])

        def transform(input):
            return input*0.5 + 2

        scores = transform(numpy.max(probabilities, axis=1))
        adversarial_scores = transform(numpy.max(adversarial_probabilities, axis=2))
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0.5, errors=None,
                                                 include_misclassifications=False, detector=None, clean_scores=scores,
                                                 adversarial_scores=adversarial_scores)
        self.assertAlmostEqual(eval.test_error(), 0)
        self.assertAlmostEqual(eval.robust_test_error(), 1)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, transform(0))
        self.assertAlmostEqual(eval.confidence_at_tpr(0.2), transform(0.9))
        self.assertAlmostEqual(eval.confidence_at_tpr(0.4), transform(0.8))
        self.assertAlmostEqual(eval.confidence_at_tpr(0.6), transform(0.7))
        self.assertAlmostEqual(eval.confidence_at_tpr(0.8), transform(0.6))
        self.assertAlmostEqual(eval.confidence_at_tpr(1), transform(0.5))

        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

    def testAdversarialEvaluationAttemptsNoValidation(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
        ])
        adversarial_probabilities = numpy.array([
            [
                self.distributionAt(9, 0.9),
                self.distributionAt(0, 0.5),
                self.distributionAt(1, 0.6),
                self.distributionAt(2, 0.7),
                self.distributionAt(3, 0.8),
                self.distributionAt(4, 0.9),
                self.distributionAt(5, 0.5),
                self.distributionAt(6, 0.6),
                self.distributionAt(7, 0.7),
                self.distributionAt(8, 0.8),
            ],
            [
                self.distributionAt(9, 0.9),
                self.distributionAt(0, 0.5),
                self.distributionAt(1, 0.6),
                self.distributionAt(2, 0.7),
                self.distributionAt(3, 0.8),
                self.distributionAt(4, 0.9),
                self.distributionAt(5, 0.5),
                self.distributionAt(6, 0.6),
                self.distributionAt(7, 0.7),
                self.distributionAt(8, 0.8),
            ]
        ])
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        test_probabilities = numpy.tile(probabilities, (2, 1))
        numpy.testing.assert_almost_equal(test_probabilities, eval.reference_probabilities)
        test_labels = numpy.tile(labels, 2)
        numpy.testing.assert_almost_equal(test_labels, eval.reference_labels)

    def testAdversarialEvaluationAttemptsNoValidationTrivialScores(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
        ])
        adversarial_probabilities = numpy.array([
            [
                self.distributionAt(9, 0.9),
                self.distributionAt(0, 0.5),
                self.distributionAt(1, 0.6),
                self.distributionAt(2, 0.7),
                self.distributionAt(3, 0.8),
                self.distributionAt(4, 0.9),
                self.distributionAt(5, 0.5),
                self.distributionAt(6, 0.6),
                self.distributionAt(7, 0.7),
                self.distributionAt(8, 0.8),
            ],
            [
                self.distributionAt(9, 0.9),
                self.distributionAt(0, 0.5),
                self.distributionAt(1, 0.6),
                self.distributionAt(2, 0.7),
                self.distributionAt(3, 0.8),
                self.distributionAt(4, 0.9),
                self.distributionAt(5, 0.5),
                self.distributionAt(6, 0.6),
                self.distributionAt(7, 0.7),
                self.distributionAt(8, 0.8),
            ]
        ])
        scores = numpy.max(probabilities, axis=1)
        adversarial_scores = numpy.max(adversarial_probabilities, axis=2)
        eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels, validation=0, errors=None,
                                                 include_misclassifications=False, detector=None, clean_scores=scores,
                                                 adversarial_scores=adversarial_scores)
        test_probabilities = numpy.tile(probabilities, (2, 1))
        numpy.testing.assert_almost_equal(test_probabilities, eval.reference_probabilities)
        test_labels = numpy.tile(labels, 2)
        numpy.testing.assert_almost_equal(test_labels, eval.reference_labels)

    def testDistalEvaluationNotCorrectShape(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        distal_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.DistalEvaluation, probabilities, distal_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        distal_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.DistalEvaluation, probabilities, distal_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        distal_probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.assertRaises(AssertionError, common.eval.DistalEvaluation, probabilities, distal_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        distal_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.DistalEvaluation, probabilities, distal_probabilities, labels)

    def testDistalEvaluationCorrectShapes(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        distal_probabilities = numpy.array([
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ])
        eval = common.eval.DistalEvaluation(probabilities, distal_probabilities, labels, validation=0)
        self.assertEqual(eval.test_probabilities.shape[0], 3 * 10)
        self.assertEqual(eval.test_confidences.shape[0], 3 * 10)
        self.assertEqual(eval.test_distal_probabilities.shape[0], 3 * 10)
        self.assertEqual(eval.test_distal_confidences.shape[0], 3 * 10)

    def testDistalEvaluationCorrectShapesWithErrors(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        distal_probabilities = numpy.array([
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ])
        errors = numpy.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
        eval = common.eval.DistalEvaluation(probabilities, distal_probabilities, labels, validation=0, errors=errors)
        self.assertEqual(eval.test_probabilities.shape[0], 10)
        self.assertEqual(eval.test_confidences.shape[0], 10)
        self.assertEqual(eval.test_distal_probabilities.shape[0], 10)
        self.assertEqual(eval.test_distal_confidences.shape[0], 10)

        errors = numpy.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        eval = common.eval.DistalEvaluation(probabilities, distal_probabilities, labels, validation=0, errors=errors)
        self.assertEqual(eval.test_probabilities.shape[0], 10)
        self.assertEqual(eval.test_confidences.shape[0], 10)
        self.assertEqual(eval.test_distal_probabilities.shape[0], 10)
        self.assertEqual(eval.test_distal_confidences.shape[0], 10)

    def testDistalEvaluationNotProbabilities(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09],
            [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        distal_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.DistalEvaluation, probabilities, labels, distal_probabilities)

        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        distal_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09],
            [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.DistalEvaluation, probabilities, labels, distal_probabilities)

    def testDistalEvaluationTPRAndFPR(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
        ])
        distal_probabilities = numpy.array([[
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
        ]])
        eval = common.eval.DistalEvaluation(probabilities, distal_probabilities, labels, validation=0.5)
        self.assertAlmostEqual(eval.test_error(), 0)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.2), 0.9)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.4), 0.8)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.6), 0.7)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.8), 0.6)
        self.assertAlmostEqual(eval.confidence_at_tpr(1), 0.5)

        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

    def testDistalEvaluationTPRAndFPRScores(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
        ])
        distal_probabilities = numpy.array([[
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
        ]])
        scores = numpy.max(probabilities, axis=1)
        distal_scores = numpy.max(distal_probabilities, axis=2)
        eval = common.eval.DistalEvaluation(probabilities, distal_probabilities, labels, validation=0.5,
                                            include_misclassifications=False, detector=None,
                                            clean_scores=scores, distal_scores=distal_scores)
        self.assertAlmostEqual(eval.test_error(), 0)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.2), 0.9)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.4), 0.8)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.6), 0.7)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.8), 0.6)
        self.assertAlmostEqual(eval.confidence_at_tpr(1), 0.5)

        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

    def testCorruptedEvaluationNotCorrectShape(self):
        labels = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        probabilities = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.CorruptedEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        adversarial_probabilities = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.CorruptedEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.CorruptedEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        adversarial_probabilities = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.CorruptedEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.CorruptedEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.assertRaises(AssertionError, common.eval.CorruptedEvaluation, probabilities, adversarial_probabilities, labels)

    def testCorruptedEvaluationCorrectShapes(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ])
        eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.reference_labels.shape[0], 3*10)
        self.assertEqual(eval.reference_probabilities.shape[0], 3 * 10)
        self.assertEqual(eval.reference_confidences.shape[0], 3 * 10)
        self.assertEqual(eval.reference_errors.shape[0], 3 * 10)
        self.assertEqual(eval.test_corrupted_probabilities.shape[0], 3 * 10)
        self.assertEqual(eval.test_corrupted_confidences.shape[0], 3 * 10)
        self.assertEqual(eval.test_corrupted_errors.shape[0], 3 * 10)
        numpy.testing.assert_array_equal(eval.reference_predictions, numpy.tile(labels, 3))
        self.assertEqual(numpy.sum(eval.reference_errors), 0)
        self.assertEqual(numpy.sum(eval.test_corrupted_errors), 30 - 3)

    def testCorruptedEvaluationNotCorrectClasses(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.assertRaises(AssertionError, common.eval.CorruptedEvaluation, probabilities, adversarial_probabilities, labels)

    def testCorruptedEvaluationNotProbabilities(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09],
            [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.CorruptedEvaluation, probabilities, adversarial_probabilities, labels)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09],
            [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        self.assertRaises(AssertionError, common.eval.CorruptedEvaluation, probabilities, adversarial_probabilities, labels)

    def testCorruptedEvaluationTestErrorNoValidation(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # ok
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # ok
            [0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], # slightly different
            [0.099, 0.099, 0.099, 0.109, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099], # hard
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ok
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # ok
            [0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # slightly different
            [0.099, 0.099, 0.099, 0.109, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099],  # hard
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]])
        eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.test_N, probabilities.shape[0])
        self.assertEqual(eval.reference_AN, 10)
        self.assertEqual(eval.reference_A, 1)
        self.assertEqual(eval.reference_N, 10)
        self.assertEqual(eval.test_error(), 0)

        test_errors = [
            1,
            7,
            99,
            73,
        ]

        for test_error in test_errors:
            labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            labels = numpy.tile(labels, 10)
            probabilities = common.numpy.one_hot(labels, 10)
            adversarial_probabilities = numpy.array([common.numpy.one_hot(labels, 10)])

            indices = numpy.array(random.sample(range(100), test_error))
            self.assertEqual(numpy.unique(indices).shape[0], indices.shape[0])

            for i in indices:
                probabilities[i] = numpy.flip(probabilities[i])

            eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
            self.assertEqual(eval.test_error(), test_error/100)
            self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

            for threshold in numpy.linspace(0, 1, 50):
                self.assertEqual(eval.test_error_at_confidence(threshold), test_error/100)

    def testCorruptedEvaluationCorruptedTestErrorNoValidation(self):
        corrupted_test_errors = [
            1,
            7,
            99,
            73,
        ]

        for corrupted_test_error in corrupted_test_errors:
            labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            labels = numpy.tile(labels, 10)
            probabilities = common.numpy.one_hot(labels, 10)
            adversarial_probabilities = numpy.array([common.numpy.one_hot(labels, 10)])

            indices = numpy.array(random.sample(range(100), corrupted_test_error))
            self.assertEqual(numpy.unique(indices).shape[0], indices.shape[0])

            for i in indices:
                adversarial_probabilities[0][i] = numpy.flip(adversarial_probabilities[0][i])

            eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
            self.assertEqual(eval.test_error(), 0)
            self.assertEqual(eval.corrupted_test_error(), corrupted_test_error/100.)
            # here all regular images are classified correctly!
            self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

            for threshold in numpy.linspace(0, 1, 50):
                self.assertEqual(eval.test_error_at_confidence(threshold), 0)
                self.assertEqual(eval.corrupted_test_error_at_confidence(threshold), corrupted_test_error/100.)

    def testCorruptedEvaluationCorruptedTestErrorAtNoValidation(self):
        # simple case where test example confidences always higher than adversarial example confidences
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(1, 0.4),
            self.distributionAt(2, 0.45),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.55),
            self.distributionAt(5, 0.6),
            self.distributionAt(6, 0.65),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.75),
            self.distributionAt(9, 0.8),
            self.distributionAt(0, 0.85),
        ]])
        eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.test_error(), 0)
        self.assertEqual(eval.corrupted_test_error(), 1)

        self.assertEqual(eval.test_error_at_confidence(0), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0), 1)
        self.assertEqual(eval.test_error_at_confidence(0.4), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.4), 1)
        self.assertEqual(eval.test_error_at_confidence(0.45), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.45), 1)
        self.assertEqual(eval.test_error_at_confidence(0.5), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.5), 1)
        self.assertEqual(eval.test_error_at_confidence(0.55), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.55), 1)
        self.assertEqual(eval.test_error_at_confidence(0.6), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.6), 1)
        self.assertEqual(eval.test_error_at_confidence(0.65), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.65), 1)
        self.assertEqual(eval.test_error_at_confidence(0.7), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.7), 1)
        self.assertEqual(eval.test_error_at_confidence(0.75), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.75), 1)
        self.assertEqual(eval.test_error_at_confidence(0.8), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.8), 1)
        self.assertEqual(eval.test_error_at_confidence(0.85), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.85), 1)
        self.assertEqual(eval.test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.test_error_at_confidence(0.95), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.95), 0)

        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

        # more complex case where with certain threshold half of the test examples "fall out"
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.7),
            self.distributionAt(1, 0.7),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.7),
            self.distributionAt(4, 0.7),
            self.distributionAt(5, 0.7),
            self.distributionAt(6, 0.7),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.7),
            self.distributionAt(9, 0.7),
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(1, 0.4),
            self.distributionAt(2, 0.45),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.55),
            self.distributionAt(5, 0.6),
            self.distributionAt(6, 0.65),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.75),
            self.distributionAt(9, 0.8),
            self.distributionAt(0, 0.85),
        ]])
        eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.test_error(), 0)
        self.assertEqual(eval.corrupted_test_error(), 1)

        self.assertEqual(eval.test_error_at_confidence(0), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0), 1)
        self.assertEqual(eval.test_error_at_confidence(0.4), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.4), 1)
        self.assertEqual(eval.test_error_at_confidence(0.45), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.45), 1)
        self.assertEqual(eval.test_error_at_confidence(0.5), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.5), 1)
        self.assertEqual(eval.test_error_at_confidence(0.55), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.55), 1)
        self.assertEqual(eval.test_error_at_confidence(0.6), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.6), 1)
        self.assertEqual(eval.test_error_at_confidence(0.65), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.65), 1)
        self.assertEqual(eval.test_error_at_confidence(0.7), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.7), 1)
        # cases where denominator is zero for test error
        self.assertEqual(eval.test_error_at_confidence(0.75), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.75), 1)
        self.assertEqual(eval.test_error_at_confidence(0.8), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.8), 1)
        self.assertEqual(eval.test_error_at_confidence(0.85), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.85), 1)
        # cases where denominator is zero for ROBUST test error
        self.assertEqual(eval.test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.9), 0)
        self.assertEqual(eval.test_error_at_confidence(0.95), 0)
        self.assertEqual(eval.corrupted_test_error_at_confidence(0.95), 0)

        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

    def testCorruptedEvaluationROCAUCNoValidation(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.7),
            self.distributionAt(1, 0.7),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.7),
            self.distributionAt(4, 0.7),
            self.distributionAt(5, 0.7),
            self.distributionAt(6, 0.7),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.7),
            self.distributionAt(9, 0.7),
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(1, 0.5),
            self.distributionAt(2, 0.5),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.5),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.5),
            self.distributionAt(7, 0.5),
            self.distributionAt(8, 0.5),
            self.distributionAt(9, 0.5),
            self.distributionAt(0, 0.5),
        ]])
        eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.test_error(), 0)
        self.assertEqual(eval.corrupted_test_error(), 1)
        self.assertEqual(eval.receiver_operating_characteristic_auc(), 1)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(1, 0.5),
            self.distributionAt(2, 0.5),
            self.distributionAt(3, 0.5),
            self.distributionAt(4, 0.5),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.5),
            self.distributionAt(7, 0.5),
            self.distributionAt(8, 0.5),
            self.distributionAt(9, 0.5),
            self.distributionAt(0, 0.5),

        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(0, 0.7),
            self.distributionAt(1, 0.7),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.7),
            self.distributionAt(4, 0.7),
            self.distributionAt(5, 0.7),
            self.distributionAt(6, 0.7),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.7),
            self.distributionAt(9, 0.7),
        ]])
        eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertEqual(eval.test_error(), 1)
        self.assertEqual(eval.corrupted_test_error(), 0)
        self.assertEqual(eval.receiver_operating_characteristic_auc(), 0)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0.1)

    def testCorruptedEvaluationTPRAndFPR(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
        ])
        adversarial_probabilities = numpy.array([[
            self.distributionAt(9, 0.9),
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
        ]])
        eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0.5)
        self.assertAlmostEqual(eval.test_error(), 0)
        self.assertAlmostEqual(eval.corrupted_test_error(), 1)
        self.assertRaises(AssertionError, eval.confidence_at_tpr, 0)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.2), 0.9)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.4), 0.8)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.6), 0.7)
        self.assertAlmostEqual(eval.confidence_at_tpr(0.8), 0.6)
        self.assertAlmostEqual(eval.confidence_at_tpr(1), 0.5)

        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.tpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(1)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.99)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.9)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.81)), 1)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.8)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.79)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.7)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.61)), 0.8)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.6)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.59)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.5)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.41)), 0.6)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.4)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.39)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.3)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.21)), 0.4)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.2)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.19)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.1)), 0.2)
        self.assertAlmostEqual(eval.fpr_at_confidence(eval.confidence_at_tpr(0.01)), 0.2)

    def testCorruptedEvaluationAttemptsNoValidation(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
            self.distributionAt(9, 0.9),
        ])
        adversarial_probabilities = numpy.array([
            [
                self.distributionAt(9, 0.9),
                self.distributionAt(0, 0.5),
                self.distributionAt(1, 0.6),
                self.distributionAt(2, 0.7),
                self.distributionAt(3, 0.8),
                self.distributionAt(4, 0.9),
                self.distributionAt(5, 0.5),
                self.distributionAt(6, 0.6),
                self.distributionAt(7, 0.7),
                self.distributionAt(8, 0.8),
            ],
            [
                self.distributionAt(9, 0.9),
                self.distributionAt(0, 0.5),
                self.distributionAt(1, 0.6),
                self.distributionAt(2, 0.7),
                self.distributionAt(3, 0.8),
                self.distributionAt(4, 0.9),
                self.distributionAt(5, 0.5),
                self.distributionAt(6, 0.6),
                self.distributionAt(7, 0.7),
                self.distributionAt(8, 0.8),
            ]
        ])
        eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        test_probabilities = numpy.tile(probabilities, (2, 1))
        numpy.testing.assert_almost_equal(test_probabilities, eval.reference_probabilities)
        test_labels = numpy.tile(labels, 2)
        numpy.testing.assert_almost_equal(test_labels, eval.reference_labels)

    def testCorruptedEvaluationCorrectedTestErrorWithLargeCleanTestError(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.6),
            self.distributionAt(2, 0.7),
            self.distributionAt(3, 0.8),
            self.distributionAt(4, 0.9),
            # !
            self.distributionAt(9, 0.9),
            self.distributionAt(5, 0.5),
            self.distributionAt(6, 0.6),
            self.distributionAt(7, 0.7),
            self.distributionAt(8, 0.8),
        ])
        adversarial_probabilities = numpy.array([
            [
                self.distributionAt(9, 0.9),
                self.distributionAt(0, 0.5),
                self.distributionAt(1, 0.6),
                self.distributionAt(2, 0.7),
                self.distributionAt(3, 0.8),
                self.distributionAt(4, 0.9),
                self.distributionAt(5, 0.5),
                self.distributionAt(6, 0.6),
                self.distributionAt(7, 0.7),
                self.distributionAt(8, 0.8),
            ],
            [
                self.distributionAt(0, 0.5),
                self.distributionAt(1, 0.6),
                self.distributionAt(2, 0.7),
                self.distributionAt(3, 0.8),
                self.distributionAt(4, 0.9),
                self.distributionAt(5, 0.5),
                self.distributionAt(6, 0.6),
                self.distributionAt(7, 0.7),
                self.distributionAt(8, 0.8),
                self.distributionAt(9, 0.9),
            ]
        ])
        eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0)
        self.assertAlmostEqual(eval.corrupted_test_error(), 0.5)
        self.assertAlmostEqual(eval.test_error(), 0.5)

    def testCorruptedEvaluationCorrectAveragedMetrics(self):
        labels = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probabilities = numpy.array([
            self.distributionAt(0, 0.5),
            self.distributionAt(1, 0.5),
            self.distributionAt(2, 1),
            self.distributionAt(3, 1),
            self.distributionAt(4, 1),
            self.distributionAt(5, 1),
            self.distributionAt(6, 1),
            self.distributionAt(7, 1),
            self.distributionAt(8, 1),
            self.distributionAt(9, 1),
        ])
        adversarial_probabilities1 = numpy.array([
            self.distributionAt(0, 0.4),
            self.distributionAt(1, 0.4),
            self.distributionAt(2, 0.4),
            self.distributionAt(3, 0.4),
            self.distributionAt(4, 0.4),
            # 5/10
            self.distributionAt(9, 0.9),
            self.distributionAt(5, 0.9),
            self.distributionAt(6, 0.9),
            self.distributionAt(7, 0.9),
            self.distributionAt(8, 0.9),
        ])
        adversarial_probabilities2 = numpy.array([
            self.distributionAt(0, 0.4),
            self.distributionAt(1, 0.4),
            self.distributionAt(2, 0.4),
            # 7/10
            self.distributionAt(9, 0.9),
            self.distributionAt(3, 0.9),
            self.distributionAt(4, 0.9),
            self.distributionAt(5, 0.9),
            self.distributionAt(6, 0.9),
            self.distributionAt(7, 0.9),
            self.distributionAt(8, 0.9),
        ])

        adversarial_probabilities = numpy.concatenate((adversarial_probabilities1, adversarial_probabilities2), axis=0)
        adversarial_probabilities = adversarial_probabilities.reshape(2, -1, 10)

        eval1 = common.eval.CorruptedEvaluation(probabilities, numpy.expand_dims(adversarial_probabilities1, axis=0), labels, validation=0)
        eval2 = common.eval.CorruptedEvaluation(probabilities, numpy.expand_dims(adversarial_probabilities2, axis=0), labels, validation=0)
        eval = common.eval.CorruptedEvaluation(probabilities, adversarial_probabilities, labels, validation=0)

        self.assertAlmostEqual(eval1.corrupted_test_error(), 0.5)
        self.assertAlmostEqual(eval1.test_error(), 0)

        self.assertAlmostEqual(eval2.corrupted_test_error(), 0.7)
        self.assertAlmostEqual(eval2.test_error(), 0)

        self.assertAlmostEqual(eval.corrupted_test_error(), 0.6)
        self.assertAlmostEqual(eval.test_error(), 0)


if __name__ == '__main__':
    unittest.main()
