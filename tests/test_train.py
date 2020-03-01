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
import attacks
import torch
import torch.utils.data
from imgaug import augmenters as iaa


class TestTrainMNIST(unittest.TestCase):
    def setUp(self):
        self.trainset = torch.utils.data.DataLoader(common.datasets.MNISTTrainSet(indices=range(10000)), batch_size=100, shuffle=True, num_workers=4)
        self.testset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(), batch_size=100, shuffle=False, num_workers=4)
        self.adversarialset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(indices=range(1000)), batch_size=100, shuffle=False, num_workers=4)
        self.randomset = torch.utils.data.DataLoader(common.datasets.RandomTestSet(1000, size=(28, 28, 1)), batch_size=100, shuffle=False, num_workers=4)

        self.cuda = True
        self.model = models.ResNet(10, [1, 28, 28], channels=12, blocks=[3, 3, 3])
        if self.cuda:
            self.model = self.model.cuda()

    def testNormalTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        trainer = common.train.NormalTraining(self.model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 2
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF7PObjective()

        distal_attack = attacks.BatchGradientDescent()
        distal_attack.max_iterations = 2
        distal_attack.base_lr = 0.1
        distal_attack.momentum = 0
        distal_attack.c = 0
        distal_attack.lr_factor = 1.5
        distal_attack.normalized = True
        distal_attack.backtrack = True
        distal_attack.initialization = attacks.initializations.RandomInitializations([
            attacks.initializations.LInfUniformNormInitialization(epsilon),  # !
            attacks.initializations.SequentialInitializations([
                attacks.initializations.LInfUniformNormInitialization(epsilon),  # !
                attacks.initializations.SmoothInitialization()
            ])
        ])
        distal_attack.norm = attacks.norms.LInfNorm()
        distal_attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        distal_objective = attacks.objectives.UntargetedF0Objective(loss=common.torch.max_log_loss)

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(self.model, self.adversarialset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval = common.eval.AdversarialEvaluation(probabilities[:len(self.adversarialset.dataset)], adversarial_probabilities, self.adversarialset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.8, eval.receiver_operating_characteristic_auc())

        distal_perturbations, distal_probabilities, _ = common.test.attack(self.model, self.randomset, distal_attack, distal_objective, attempts=1, writer=writer,
                                                                           cuda=self.cuda)
        eval = common.eval.DistalEvaluation(probabilities[:len(self.adversarialset.dataset)], distal_probabilities, self.adversarialset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.8, eval.receiver_operating_characteristic_auc())

    def testNormalTrainingAugmentation(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=(0, 0.025)),
            iaa.Add((-0.075, 0.075)),
            common.imgaug.Clip(0, 1)
        ])

        trainer = common.train.NormalTraining(self.model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.03, eval.test_error())

    def testReferenceAdversarialTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        epsilon = 0.3
        attack = attacks.BatchReferencePGD()
        attack.max_iterations = 10
        attack.epsilon = epsilon
        attack.base_lr = 0.1
        attack.norm = attacks.norms.LInfNorm()
        objective = attacks.objectives.UntargetedF0Objective()

        trainer = common.train.AdversarialTraining(self.model, self.trainset, self.testset, optimizer, scheduler, attack, objective, fraction=1, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(self.model, self.adversarialset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval = common.eval.AdversarialEvaluation(probabilities[:len(self.adversarialset.dataset)], adversarial_probabilities, self.adversarialset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.2, eval.robust_test_error())

    def testAdversarialTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 2
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF0Objective()

        trainer = common.train.AdversarialTraining(self.model, self.trainset, self.testset, optimizer, scheduler, attack, objective, fraction=0.5, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(self.model, self.adversarialset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval = common.eval.AdversarialEvaluation(probabilities[:len(self.adversarialset.dataset)], adversarial_probabilities, self.adversarialset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.2, eval.robust_test_error())

    def testAdversarialTrainingFraction(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 2
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF0Objective()

        trainer = common.train.AdversarialTraining(self.model, self.trainset, self.testset, optimizer, scheduler, attack, objective, fraction=1, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(self.model, self.adversarialset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval = common.eval.AdversarialEvaluation(probabilities[:len(self.adversarialset.dataset)], adversarial_probabilities, self.adversarialset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.15, eval.robust_test_error())

    def testConfidenceCalibratedAdversarialTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 2
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF7PObjective()

        loss = common.torch.cross_entropy_divergence
        transition = common.utils.partial(common.torch.linear_transition, norm=attack.norm)

        trainer = common.train.ConfidenceCalibratedAdversarialTraining(self.model, self.trainset, self.testset, optimizer, scheduler, attack, objective, loss, transition, fraction=0.5, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(self.model, self.adversarialset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval = common.eval.AdversarialEvaluation(probabilities[:len(self.adversarialset.dataset)], adversarial_probabilities, self.adversarialset.dataset.labels, validation=0)
        self.assertGreaterEqual(eval.receiver_operating_characteristic_auc(), 0.95)

    def testConfidenceCalibratedAdversarialTrainingFraction(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 2
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF7PObjective()

        loss = common.torch.cross_entropy_divergence
        transition = common.utils.partial(common.torch.linear_transition, norm=attack.norm)

        self.assertRaises(AssertionError, common.train.ConfidenceCalibratedAdversarialTraining, self.model, self.trainset, self.testset, optimizer, scheduler, attack, objective, loss, transition, fraction=1, augmentation=augmentation, writer=writer, cuda=self.cuda)


class TestTrainSVHN(TestTrainMNIST):
    def setUp(self):
        self.trainset = torch.utils.data.DataLoader(common.datasets.SVHNTrainSet(indices=range(10000)), batch_size=100, shuffle=True, num_workers=4)
        self.testset = torch.utils.data.DataLoader(common.datasets.SVHNTestSet(), batch_size=100, shuffle=False, num_workers=4)
        self.adversarialset = torch.utils.data.DataLoader(common.datasets.SVHNTestSet(indices=range(1000)), batch_size=100, shuffle=False, num_workers=4)
        self.randomset = torch.utils.data.DataLoader(common.datasets.RandomTestSet(1000, size=(28, 28, 1)), batch_size=100, shuffle=False, num_workers=4)

        self.cuda = True
        self.model = models.ResNet(10, [3, 32, 32], channels=12, blocks=[3, 3, 3])
        if self.cuda:
            self.model = self.model.cuda()


if __name__ == '__main__':
    unittest.main()
