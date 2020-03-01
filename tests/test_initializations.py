import unittest
import torch
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.test
import common.train
import common.datasets
import common.torch
import common.state
import common.eval
import attacks
import torch
import torch.utils.data
import math


class TestInitializations(unittest.TestCase):
    def testRandomInitializations(self):
        epsilon = 0.3
        images = None
        initializer = attacks.initializations.RandomInitializations([
            attacks.initializations.LInfUniformNormInitialization(epsilon),
            attacks.initializations.ZeroInitialization(),
        ])

        zero = 0
        for i in range(1000):
            perturbations = torch.from_numpy(numpy.random.uniform(0, 1, size=[100, 1, 32, 32]))
            perturbations = torch.autograd.Variable(perturbations, requires_grad=True)
            initializer(images, perturbations)

            perturbations = perturbations.detach().cpu().numpy()
            if numpy.allclose(perturbations, 0):
                zero += 1
            else:
                norms = numpy.linalg.norm(perturbations.reshape(100, -1), ord=float('inf'), axis=1)
                self.assertTrue(numpy.allclose(norms, 0.3))

        self.assertGreater(zero/1000., 0.4)
        self.assertGreater(0.6, zero/1000.)

    def testGaussianInitialization(self):
        epsilon = 0.3
        images = None
        initializer = attacks.initializations.GaussianInitialization(0.3)

        perturbations_ = None
        for i in range(100):
            perturbations = torch.from_numpy(numpy.zeros([100, 1, 32, 32]))
            perturbations = torch.autograd.Variable(perturbations, requires_grad=True)
            initializer(images, perturbations)
            if perturbations_ is None:
                perturbations_ = perturbations.detach().cpu().numpy()
            else:
                perturbations_ = numpy.concatenate((perturbations_, perturbations.detach().cpu().numpy()))

        mean = numpy.mean(perturbations_)
        std = numpy.std(perturbations_)
        self.assertAlmostEqual(mean, 0, places=4)
        self.assertAlmostEqual(std, epsilon / (2*math.log(perturbations_.shape[1]*perturbations_.shape[2]*perturbations_.shape[3])), places=4)

    def testZeroInitialization(self):
        images = None
        perturbations = torch.from_numpy(numpy.random.uniform(0, 1, size=[100, 1, 32, 32]))
        perturbations = torch.autograd.Variable(perturbations, requires_grad=True)

        initializer = attacks.initializations.ZeroInitialization()
        initializer(images, perturbations)

        perturbations = perturbations.detach().cpu().numpy()
        norms = numpy.linalg.norm(perturbations.reshape(100, -1), ord=float('inf'), axis=1)
        self.assertTrue(numpy.allclose(perturbations, 0))
        self.assertAlmostEqual(numpy.sum(perturbations), 0)
        self.assertTrue(numpy.all(norms <= 0))

    def testLInfInitializations(self):
        epsilon = 0.3
        images = None
        perturbations = torch.from_numpy(numpy.zeros([100, 1, 32, 32]))
        perturbations = torch.autograd.Variable(perturbations, requires_grad=True)

        initializer = attacks.initializations.LInfUniformInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=float('inf'), axis=1)
        self.assertTrue(numpy.all(norms <= epsilon))

        initializer = attacks.initializations.LInfUniformNormInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=float('inf'), axis=1)
        self.assertTrue(numpy.all(norms <= epsilon))

        initializer = attacks.initializations.LInfUniformVolumeInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=float('inf'), axis=1)
        self.assertTrue(numpy.all(norms <= epsilon))

    def testL2Initializations(self):
        epsilon = 0.3
        images = None
        perturbations = torch.from_numpy(numpy.zeros([100, 1, 32, 32]))
        perturbations = torch.autograd.Variable(perturbations, requires_grad=True)

        initializer = attacks.initializations.L2UniformNormInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=2, axis=1)
        self.assertTrue(numpy.all(norms <= epsilon))

        initializer = attacks.initializations.L2UniformVolumeInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=2, axis=1)
        self.assertTrue(numpy.all(norms <= epsilon))

    def testL1Initializations(self):
        epsilon = 0.3
        images = None
        perturbations = torch.from_numpy(numpy.zeros([100, 1, 32, 32]))
        perturbations = torch.autograd.Variable(perturbations, requires_grad=True)

        initializer = attacks.initializations.L1UniformNormInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=1, axis=1)
        self.assertTrue(numpy.all(norms <= epsilon))

        initializer = attacks.initializations.L1UniformVolumeInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=1, axis=1)
        self.assertTrue(numpy.all(norms <= epsilon))


if __name__ == '__main__':
    unittest.main()
