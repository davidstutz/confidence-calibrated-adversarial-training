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
    def testL1Norm(self):
        pass

    def testL1NormNormalize(self):
        norm = attacks.norms.L1Norm(fraction=0.5)

        # [[[[0.0240, 0.2328],
        #  [0.0012, 0.5750]]],
        #
        #  [[[0.7762, 0.2198],
        #  [0.2525, 0.9873]]]]

        original_gradients = torch.from_numpy(numpy.array([[[[0.0240, 0.2328], [0.0012, 0.5750]]], [[[0.7762, 0.2198], [0.2525, 0.9873]]]]).astype(numpy.float32))
        normalized_gradients = original_gradients.clone()

        norm.normalize(normalized_gradients)
        self.assertAlmostEqual(normalized_gradients[0, 0, 0, 0].item(), 0)
        self.assertAlmostEqual(normalized_gradients[0, 0, 1, 0].item(), 0)
        self.assertAlmostEqual(normalized_gradients[0, 0, 0, 1].item(), original_gradients[0, 0, 0, 1].item()/(0.2328 + 0.5750))
        self.assertAlmostEqual(normalized_gradients[0, 0, 1, 1].item(), original_gradients[0, 0, 1, 1].item()/(0.2328 + 0.5750))

        self.assertAlmostEqual(normalized_gradients[1, 0, 0, 1].item(), 0)
        self.assertAlmostEqual(normalized_gradients[1, 0, 1, 0].item(), 0)
        self.assertAlmostEqual(normalized_gradients[1, 0, 0, 0].item(), original_gradients[1, 0, 0, 0].item()/(0.7762 + 0.9873))
        self.assertAlmostEqual(normalized_gradients[1, 0, 1, 1].item(), original_gradients[1, 0, 1, 1].item()/(0.7762 + 0.9873))


if __name__ == '__main__':
    unittest.main()
