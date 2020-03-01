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


class TestProjections(unittest.TestCase):
    def testLInfProjection(self):
        epsilon = 0.3
        perturbations = torch.from_numpy(numpy.zeros([100, 1, 32, 32]).astype(numpy.float32))
        perturbations = torch.autograd.Variable(perturbations, requires_grad=True)

        images = torch.from_numpy(numpy.random.uniform(0, 1, size=(100, 1, 32, 32)).astype(numpy.float32))
        images = torch.autograd.Variable(images)

        initializer = attacks.initializations.UniformInitialization()
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=float('inf'), axis=1)
        self.assertTrue(numpy.any(norms > epsilon), 'max norm=%g' % numpy.max(norms))

        projection = attacks.projections.LInfProjection(epsilon)
        projection(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=float('inf'), axis=1)
        self.assertTrue(numpy.all(norms <= epsilon + 1e-6), 'max norm=%g' % numpy.max(norms))

    def testL1Projection(self):
        epsilon = 0.3
        perturbations = torch.from_numpy(numpy.zeros([100, 1, 32, 32]).astype(numpy.float32))
        perturbations = torch.autograd.Variable(perturbations, requires_grad=True)

        images = torch.from_numpy(numpy.random.uniform(0, 1, size=(100, 1, 32, 32)).astype(numpy.float32))
        images = torch.autograd.Variable(images)

        initializer = attacks.initializations.UniformInitialization()
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=1, axis=1)
        self.assertTrue(numpy.any(norms > epsilon), 'max norm=%g' % numpy.max(norms))

        projection = attacks.projections.L1Projection(epsilon)
        projection(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=1, axis=1)
        self.assertTrue(numpy.all(norms <= epsilon + 1e-4), 'max norm=%g' % numpy.max(norms))

    def testL2Projection(self):
        epsilon = 0.3
        perturbations = torch.from_numpy(numpy.zeros([100, 1, 32, 32]).astype(numpy.float32))
        perturbations = torch.autograd.Variable(perturbations, requires_grad=True)

        images = torch.from_numpy(numpy.random.uniform(0, 1, size=(100, 1, 32, 32)).astype(numpy.float32))
        images = torch.autograd.Variable(images)

        initializer = attacks.initializations.UniformInitialization()
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=2, axis=1)
        self.assertTrue(numpy.any(norms > epsilon), 'max norm=%g' % numpy.max(norms))

        projection = attacks.projections.L2Projection(epsilon)
        projection(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=2, axis=1)
        self.assertTrue(numpy.all(norms <= epsilon + 1e-6), 'max norm=%g' % numpy.max(norms))

    def testBoxProjection(self):
        epsilon = 0.3
        perturbations = torch.from_numpy(numpy.zeros([100, 1, 32, 32]).astype(numpy.float32))
        perturbations = torch.autograd.Variable(perturbations, requires_grad=True)

        images = torch.from_numpy(numpy.random.uniform(0, 1, size=(100, 1, 32, 32)).astype(numpy.float32))
        images = torch.autograd.Variable(images)

        initializer = attacks.initializations.L2UniformNormInitialization(1)
        initializer(images, perturbations)

        perturbed_images = images.detach().cpu().numpy() + perturbations.detach().cpu().numpy()
        self.assertTrue(numpy.any(perturbed_images > 1), 'max value=%g' % numpy.max(perturbed_images))
        self.assertTrue(numpy.any(perturbed_images < 0), 'min value=%g' % numpy.max(perturbed_images))

        projection = attacks.projections.BoxProjection(epsilon)
        projection(images, perturbations)

        perturbed_images = images.detach().cpu().numpy() + perturbations.detach().cpu().numpy()
        self.assertTrue(numpy.all(perturbed_images <= 1), 'max value=%g' % numpy.max(perturbed_images))
        self.assertTrue(numpy.all(perturbed_images >= 0), 'min value=%g' % numpy.max(perturbed_images))


if __name__ == '__main__':
    unittest.main()
