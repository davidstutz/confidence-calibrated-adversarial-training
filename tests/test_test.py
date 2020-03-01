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
import common.imgaug
import common.eval
import common.summary
import torch
import attacks
import torch.utils.data
from imgaug import augmenters as iaa


class TestTest(unittest.TestCase):
    def setUp(self):
        self.trainset = torch.utils.data.DataLoader(common.datasets.MNISTTrainSet(indices=range(10000)), batch_size=100, shuffle=True, num_workers=4)
        self.testset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(), batch_size=100, num_workers=4, shuffle=False)
        self.adversarialset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(indices=range(1000)), batch_size=100, num_workers=4, shuffle=False)
        self.cuda = True

    def testTest(self):
        model = models.LeNet(10, [1, 28, 28], channels=12)

        if self.cuda:
            model = model.cuda()

        model.eval()
        probabilities = common.test.test(model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels)
        self.assertGreaterEqual(0.05, abs(0.9 - eval.test_error()))

    def testAttack(self):
        model = models.LeNet(10, [1, 28, 28], channels=12)
        #state = common.state.State.load('mnist_lenet.pth.tar')
        #model = state.model

        if self.cuda:
            model = model.cuda()

        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 2
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = True
        attack.backtrack = False
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF0Objective()

        model.eval()
        attempts = 1
        perturbations, adversarial_probabilities, errors = common.test.attack(model, self.adversarialset, attack, objective, attempts=attempts, writer=common.summary.SummaryWriter(), cuda=self.cuda)

        self.assertEqual(perturbations.shape[0], attempts)
        self.assertEqual(perturbations.shape[1], self.adversarialset.dataset.images.shape[0])
        self.assertEqual(perturbations.shape[2], self.adversarialset.dataset.images.shape[3])
        self.assertEqual(perturbations.shape[3], self.adversarialset.dataset.images.shape[1])
        self.assertEqual(perturbations.shape[4], self.adversarialset.dataset.images.shape[2])
        self.assertEqual(adversarial_probabilities.shape[0], attempts)
        self.assertEqual(adversarial_probabilities.shape[1], perturbations.shape[1])
        self.assertEqual(adversarial_probabilities.shape[2], numpy.max(self.adversarialset.dataset.labels) + 1)

        perturbations = numpy.transpose(perturbations, (0, 1, 3, 4, 2))
        adversarialloader = torch.utils.data.DataLoader(common.datasets.AdversarialDataset(self.adversarialset.dataset.images, perturbations, self.adversarialset.dataset.labels), batch_size=100, shuffle=False)
        self.assertEqual(len(adversarialloader), attempts*len(self.adversarialset))
        clean_probabilities = common.test.test(model, adversarialloader, cuda=self.cuda)

        adversarial_probabilities = adversarial_probabilities.reshape(adversarial_probabilities.shape[0]*adversarial_probabilities.shape[1], adversarial_probabilities.shape[2])
        self.assertTrue(numpy.all(numpy.sum(perturbations.reshape(perturbations.shape[0]*perturbations.shape[1], -1), axis=1) > 0))
        numpy.testing.assert_array_almost_equal(clean_probabilities, adversarial_probabilities)


if __name__ == '__main__':
    unittest.main()
