import unittest
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import common.utils
import common.datasets
import common.imgaug
from imgaug import augmenters as iaa
import attacks
import models
import math
import torch
import copy


class TestTransitions(unittest.TestCase):
    def testPowerTransition(self):
        for epsilon in [0.01, 0.05, 0.1, 0.3]:
            for pow in [1, 2, 4, 6, 8]:
                perturbations = torch.zeros(100, 3, 32, 32)
                for b in range(perturbations.size(0)):
                    perturbations[b, 0, 0, 0] = (b + 1)*epsilon/100.

                norm = attacks.norms.LInfNorm()
                transition = common.utils.partial(common.torch.power_transition, epsilon=epsilon, gamma=pow, norm=norm)
                gammas, norms = transition(perturbations)

                for b in range(perturbations.size(0) - 1):
                    self.assertGreaterEqual(gammas[b + 1].item(), gammas[b].item())

                self.assertGreaterEqual(gammas[b//2].item(), 0.5)

                for b in range(perturbations.size(0)):
                    self.assertAlmostEqual(norms[b].item(), (b + 1)*epsilon/100., places=4)
                    self.assertAlmostEqual(gammas[b].item(), 1 - (1 - min(1, norms[b].item()/epsilon))**pow, places=2)

    def testExponentialTransition(self):
        epsilon = 0.3
        for gamma in [8, 10, 12]:
            perturbations = torch.zeros(100, 3, 32, 32)
            for b in range(perturbations.size(0)):
                perturbations[b, 0, 0, 0] = (b + 1)*epsilon/100.

            norm = attacks.norms.LInfNorm()
            transition = common.utils.partial(common.torch.exponential_transition, epsilon=epsilon, gamma=gamma, norm=norm)
            gammas, norms = transition(perturbations)

            for b in range(perturbations.size(0) - 1):
                self.assertGreaterEqual(gammas[b + 1].item(), gammas[b].item())

            self.assertGreaterEqual(gammas[b//2].item(), 0.5)

            for b in range(perturbations.size(0)):
                self.assertAlmostEqual(norms[b].item(), (b + 1)*epsilon/100., places=4)
                self.assertAlmostEqual(gammas[b].item(), 1 - math.exp(-gamma*norms[b].item()), places=2)

        for gamma in [1, 2, 4]:
            perturbations = torch.zeros(100, 3, 32, 32)
            for b in range(perturbations.size(0)):
                perturbations[b, 0, 0, 0] = (b + 1) * epsilon / 100.

            norm = attacks.norms.LInfNorm()
            transition = common.utils.partial(common.torch.exponential_transition, epsilon=epsilon, gamma=gamma, norm=norm)
            gammas, norms = transition(perturbations)

            for b in range(perturbations.size(0) - 1):
                self.assertGreaterEqual(gammas[b + 1].item(), gammas[b].item())

            # !
            self.assertGreaterEqual(0.5, gammas[b // 2].item())

            for b in range(perturbations.size(0)):
                self.assertAlmostEqual(norms[b].item(), (b + 1) * epsilon / 100., places=4)
                self.assertAlmostEqual(gammas[b].item(), 1 - math.exp(-gamma * norms[b].item()), places=2)


if __name__ == '__main__':
    unittest.main()
