import unittest
import numpy
import torch
import sys
import os

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.torch


class TestTorch(unittest.TestCase):
    def testProjectBall(self):
        N = 1000
        epsilons = [4, 5, 15, 0.3]
        ords = [0, 1, 2, float('inf')]
        for i in range(len(ords)):
            ord = ords[i]
            epsilon = epsilons[i]
            for D in [2, 3, 5, 10, 100]:
                original_data = torch.from_numpy(numpy.random.randn(N, D))
                original_norms = torch.norm(original_data, p=ord, dim=1)

                projected_data = common.torch.project_ball(original_data, ord=ord, epsilon=epsilon)
                projected_norms = torch.norm(projected_data, p=ord, dim=1)

                original_norms = original_norms.numpy()
                projected_norms = projected_norms.numpy()

                evaluated1 = numpy.logical_and(original_norms <= epsilon, numpy.abs(original_norms - projected_norms) <= 0.001)
                evaluated2 = numpy.logical_and(original_norms > epsilon, numpy.abs(projected_norms - epsilon) <= 0.0001)
                self.assertTrue(numpy.all(numpy.logical_or(evaluated1, evaluated2)))


if __name__ == '__main__':
    unittest.main()
