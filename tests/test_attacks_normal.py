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


class TestAttacksNormalMNISTMLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 100
        cls.cuda = True
        cls.setDatasets()

        if os.path.exists(cls.getModelFile()):
            state = common.state.State.load(cls.getModelFile())
            cls.model = state.model

            if cls.cuda:
                cls.model = cls.model.cuda()
        else:
            cls.model = cls.getModel()
            if cls.cuda:
                cls.model = cls.model.cuda()
            print(cls.model)

            optimizer = torch.optim.SGD(cls.model.parameters(), lr=0.1, momentum=0.9)
            scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(cls.trainloader))
            writer = common.summary.SummaryWriter()
            augmentation = None

            trainer = common.train.NormalTraining(cls.model, cls.trainloader, cls.testloader, optimizer, scheduler, augmentation=augmentation, writer=writer,
                                                  cuda=cls.cuda)
            for e in range(10):
                trainer.step(e)

            common.state.State.checkpoint(cls.getModelFile(), cls.model, optimizer, scheduler, e)

            cls.model.eval()
            probabilities = common.test.test(cls.model, cls.testloader, cuda=cls.cuda)
            eval = common.eval.CleanEvaluation(probabilities, cls.testloader.dataset.labels, validation=0)
            assert 0.075 > eval.test_error(), '0.05 !> %g' % eval.test_error()
            assert numpy.mean(numpy.max(probabilities, axis=1)) > 0.9

        cls.model.eval()

    @classmethod
    def setDatasets(cls):
        cls.trainset = common.datasets.MNISTTrainSet()
        cls.testset = common.datasets.MNISTTestSet()
        cls.adversarialset = common.datasets.MNISTTestSet(indices=range(100))
        cls.trainloader = torch.utils.data.DataLoader(cls.trainset, batch_size=cls.batch_size, shuffle=True, num_workers=0)
        cls.testloader = torch.utils.data.DataLoader(cls.testset, batch_size=cls.batch_size, shuffle=False, num_workers=0)
        cls.adversarialloader = torch.utils.data.DataLoader(cls.adversarialset, batch_size=cls.batch_size, shuffle=False, num_workers=0)

    @classmethod
    def getModelFile(cls):
        return 'mnist_mlp.pth.tar'

    @classmethod
    def getModel(cls):
        return models.MLP(10, [1, 28, 28], units=[100, 100, 100], action=torch.nn.Sigmoid)

    def successRate(self, images, perturbations, labels):
        adversarialloader = torch.utils.data.DataLoader(common.datasets.AdversarialDataset(images, perturbations, labels), batch_size=100)
        testloader = torch.utils.data.DataLoader(self.adversarialset, batch_size=100, shuffle=False)
        self.assertEqual(len(adversarialloader), len(testloader))

        # assumes one attempt only!
        clean_probabilities = common.test.test(self.model, testloader, cuda=self.cuda)
        adversarial_probabilities = numpy.array([common.test.test(self.model, adversarialloader, cuda=self.cuda)])

        eval = common.eval.AdversarialEvaluation(clean_probabilities, adversarial_probabilities, labels, validation=0)
        return eval.success_rate()

    def runTestAttackProjections(self, attack, epsilon, objective=attacks.objectives.UntargetedF0Objective()):
        for b, (images, labels) in enumerate(self.testloader):
            break

        images = common.torch.as_variable(images, self.cuda).permute(0, 3, 1, 2)
        labels = common.torch.as_variable(labels, self.cuda)

        objective.set(labels)
        perturbations, errors = attack.run(self.model, images, objective)
        norms = numpy.linalg.norm(perturbations.reshape(self.batch_size, -1), ord=float('inf'), axis=1)

        perturbed_images = images.cpu().numpy() + perturbations
        self.assertTrue(numpy.all(perturbed_images <= 1 + 1e-6), 'max value=%g' % numpy.max(perturbed_images))
        self.assertTrue(numpy.all(perturbed_images >= 0 - 1e-6), 'min value=%g' % numpy.min(perturbed_images))
        self.assertTrue(numpy.all(norms <= epsilon + 1e-6), 'max norm=%g epsilon=%g' % (numpy.max(norms), epsilon))

    def runTestAttackPerformance(self, attack, attempts=5, objective=attacks.objectives.UntargetedF0Objective()):
        for b, (images, labels) in enumerate(self.adversarialloader):
            break

        images = common.torch.as_variable(images, self.cuda).permute(0, 3, 1, 2)
        labels = common.torch.as_variable(labels, self.cuda)

        success_rate = 0
        for t in range(attempts):
            objective.set(labels)
            perturbations, errors = attack.run(self.model, images, objective)

            perturbations = numpy.array([numpy.transpose(perturbations, (0, 2, 3, 1))])
            success_rate += self.successRate(numpy.transpose(images.cpu().numpy(), (0, 2, 3, 1)), perturbations, labels.cpu().numpy())

        success_rate /= attempts
        return success_rate

    def runTestDeterministicAttack(self, attack, objective=attacks.objectives.UntargetedF0Objective()):
        first = True
        epsilon = 0.1

        for b, (images, labels) in enumerate(self.testloader):
            break

        images = common.torch.as_variable(images, self.cuda).permute(0, 3, 1, 2)
        labels = common.torch.as_variable(labels, self.cuda)

        numpy_images = numpy.transpose(images.cpu().numpy().copy(), (0, 2, 3, 1))
        numpy_labels = labels.cpu().numpy().copy()

        initial_perturbations = torch.zeros_like(images)
        attacks.initializations.LInfUniformNormInitialization(epsilon)(images, initial_perturbations)

        attack.initialization = attacks.initializations.FixedInitialization(initial_perturbations)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        # attack.projection = attacks.projections.BoxProjection()
        attack.norm = attacks.norms.LInfNorm()

        objective.set(labels)

        for i in range(5):
            perturbations, errors = attack.run(self.model, images, objective)
            norms = numpy.linalg.norm(perturbations.reshape(self.batch_size, -1), ord=float('inf'), axis=1)
            self.assertTrue(numpy.all(norms <= epsilon + 1e-4), numpy.max(norms))

            perturbations = numpy.array([numpy.transpose(perturbations, (0, 2, 3, 1))])
            success_rate = self.successRate(numpy_images, perturbations, numpy_labels)
            self.assertEqual(len(perturbations.shape), 5)

            if first is True:
                first_norms = norms.copy()
                first_perturbations = perturbations.copy()
                first_success_rate = success_rate
                first = False

            numpy.testing.assert_array_almost_equal(norms, first_norms, 4)
            numpy.testing.assert_array_almost_equal(perturbations, first_perturbations, 4)
            self.assertAlmostEqual(success_rate, first_success_rate, 4)

    def testBatchReferencePGD(self):
        attack = attacks.batch_reference_pgd.BatchReferencePGD()
        attack.max_iterations = 10
        attack.epsilon = 0.3
        attack.base_lr = 0.1

        self.runTestAttackProjections(attack, attack.epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchOptimSGDStep(self):
        optimizer = torch.optim.SGD
        lr = 0.1
        momentum = 0
        attack = attacks.batch_optim.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 1
        attack.c = 0
        attack.initialization = None
        attack.projection = None
        attack.normalized = False
        attack.norm = attacks.norms.LInfNorm()

        for i in range(10):
            image, label = self.testloader.dataset[i]

            images = common.torch.as_variable(torch.from_numpy(numpy.expand_dims(image, axis=0)), self.cuda)
            images = images.permute(0, 3, 1, 2)
            labels = common.torch.as_variable(torch.from_numpy(numpy.array([label])), self.cuda)
            objective = attacks.objectives.UntargetedF0Objective()
            objective.set(labels)

            # pre-compute step
            manual_perturbations = torch.zeros_like(images)
            manual_perturbations.requires_grad = True
            manual_logits = self.model(images + manual_perturbations)
            manual_error = objective(manual_logits)
            manual_error.backward()

            manual_gradients = manual_perturbations.grad
            next_perturbations = manual_perturbations - lr * manual_gradients

            next_logits = self.model(images + next_perturbations)
            next_error = objective(next_logits)
            if next_error.item() < manual_error.item():
                manual_perturbations = next_perturbations
            manual_perturbations = manual_perturbations.detach().cpu().numpy()

            # now run attack
            attack_perturbations, _ = attack.run(self.model, images, objective)

            numpy.testing.assert_allclose(manual_perturbations, attack_perturbations, atol=1e-4)

    def testBatchOptimSGDStepNormalized(self):
        optimizer = torch.optim.SGD
        lr = 0.1
        momentum = 0
        attack = attacks.batch_optim.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 1
        attack.c = 0
        attack.initialization = None
        attack.projection = None
        attack.normalized = True
        attack.norm = attacks.norms.LInfNorm()

        for i in range(10):
            image, label = self.testloader.dataset[i]

            images = common.torch.as_variable(torch.from_numpy(numpy.expand_dims(image, axis=0)), self.cuda)
            images = images.permute(0, 3, 1, 2)
            labels = common.torch.as_variable(torch.from_numpy(numpy.array([label])), self.cuda)
            objective = attacks.objectives.UntargetedF0Objective()
            objective.set(labels)

            # pre-compute step
            manual_perturbations = torch.zeros_like(images)
            manual_perturbations.requires_grad = True
            manual_logits = self.model(images + manual_perturbations)
            manual_error = objective(manual_logits)
            manual_error.backward()

            manual_gradients = manual_perturbations.grad
            next_perturbations = manual_perturbations - lr * torch.sign(manual_gradients)

            next_logits = self.model(images + next_perturbations)
            next_error = objective(next_logits)
            if next_error.item() < manual_error.item():
                manual_perturbations = next_perturbations
            manual_perturbations = manual_perturbations.detach().cpu().numpy()

            # now run attack
            attack_perturbations, _ = attack.run(self.model, images, objective)

            numpy.testing.assert_allclose(manual_perturbations, attack_perturbations, atol=1e-4)

    def testBatchOptimSGDUnnormalized(self):
        epsilon = 0.3
        optimizer = torch.optim.SGD
        lr = 100
        momentum = 0
        attack = attacks.batch_optim.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 20
        attack.c = 0
        attack.normalized = False
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.6)

    def testBatchOptimSGDUnnormalizedZero(self):
        epsilon = 0.3
        optimizer = torch.optim.SGD
        lr = 100
        momentum = 0
        attack = attacks.batch_optim.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 20
        attack.c = 0
        attack.normalized = False
        attack.initialization = None
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.25)
        self.assertGreaterEqual(0.35, success_rate)

    def testBatchOptimSGDNormalized(self):
        epsilon = 0.3
        optimizer = torch.optim.SGD
        lr = 1
        momentum = 0
        attack = attacks.batch_optim.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 10
        attack.c = 0
        attack.normalized = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchOptimDeterministicUnnormalized(self):
        optimizer = torch.optim.SGD
        lr = 0.5
        momentum = 0
        attack = attacks.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 10
        attack.c = 0
        attack.normalized = False

        self.runTestDeterministicAttack(attack)

    def testBatchOptimDeterministicNormalized(self):
        optimizer = torch.optim.SGD
        lr = 0.5
        momentum = 0
        attack = attacks.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 10
        attack.c = 0
        attack.normalized = True

        self.runTestDeterministicAttack(attack)

    def testBatchGradientDescentStep(self):
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 1
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = False
        attack.backtrack = False
        attack.initialization = None
        attack.projection = None
        attack.norm = attacks.norms.LInfNorm()

        for i in range(10):
            image, label = self.testloader.dataset[i]

            images = common.torch.as_variable(torch.from_numpy(numpy.expand_dims(image, axis=0)), self.cuda)
            images = images.permute(0, 3, 1, 2)
            labels = common.torch.as_variable(torch.from_numpy(numpy.array([label])), self.cuda)
            objective = attacks.objectives.UntargetedF0Objective()
            objective.set(labels)

            # pre-compute step
            manual_perturbations = torch.zeros_like(images)
            manual_perturbations.requires_grad = True
            manual_logits = self.model(images + manual_perturbations)
            manual_error = objective(manual_logits)
            manual_error.backward()

            manual_gradients = manual_perturbations.grad
            next_perturbations = manual_perturbations - attack.base_lr * manual_gradients

            next_logits = self.model(images + next_perturbations)
            next_error = objective(next_logits)
            if next_error.item() < manual_error.item():
                manual_perturbations = next_perturbations
            manual_perturbations = manual_perturbations.detach().cpu().numpy()

            # now run attack
            attack_perturbations, _ = attack.run(self.model, images, objective)

            numpy.testing.assert_allclose(manual_perturbations, attack_perturbations, atol=1e-4)

    def testBatchGradientDescentStepNormalized(self):
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 1
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = True
        attack.backtrack = False
        attack.initialization = None
        attack.projection = None
        attack.norm = attacks.norms.LInfNorm()

        for i in range(10):
            image, label = self.testloader.dataset[i]

            images = common.torch.as_variable(torch.from_numpy(numpy.expand_dims(image, axis=0)), self.cuda)
            images = images.permute(0, 3, 1, 2)
            labels = common.torch.as_variable(torch.from_numpy(numpy.array([label])), self.cuda)
            objective = attacks.objectives.UntargetedF0Objective()
            objective.set(labels)

            # pre-compute step
            manual_perturbations = torch.zeros_like(images)
            manual_perturbations.requires_grad = True
            manual_logits = self.model(images + manual_perturbations)
            manual_error = objective(manual_logits)
            manual_error.backward()

            manual_gradients = manual_perturbations.grad
            next_perturbations = manual_perturbations - attack.base_lr*torch.sign(manual_gradients)

            next_logits = self.model(images + next_perturbations)
            next_error = objective(next_logits)
            if next_error.item() < manual_error.item():
                manual_perturbations = next_perturbations
            manual_perturbations = manual_perturbations.detach().cpu().numpy()

            # now run attack
            attack_perturbations, _ = attack.run(self.model, images, objective)

            numpy.testing.assert_allclose(manual_perturbations, attack_perturbations, atol=1e-4)

    def testBatchGradientDescentUnnormalized(self):
        epsilon = 0.3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 20
        attack.base_lr = 100
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = False  # !
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.6)

    def testBatchGradientDescentUnnormalizedZero(self):
        epsilon = 0.3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 20
        attack.base_lr = 100
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = False
        attack.initialization = None
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.25)
        self.assertGreaterEqual(0.35, success_rate)

    def testBatchGradientDescentNormalized(self):
        epsilon = 0.3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchGradientDescentNormalizedZero(self):
        epsilon = 0.3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = True
        attack.initialization = None
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchGradientDescentNormalizedBacktrack(self):
        epsilon = 0.3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 5
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchGradientDescentInitializationRelative(self):
        success_rates = []
        epsilon = 0.3

        # zero initialization
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 5
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = None
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rates.append(self.runTestAttackPerformance(attack))

        # uniform norm
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 5
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rates.append(self.runTestAttackPerformance(attack))

        self.assertGreaterEqual(success_rates[1], success_rates[0])
        self.assertGreaterEqual(success_rates[1], 0.95)

    def testBatchGradientDescentOptimizationRelative(self):
        success_rates = []
        epsilon = 0.3

        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 20
        attack.base_lr = 100
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rates.append(self.runTestAttackPerformance(attack))

        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.backtrack = False
        attack.normalized = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rates.append(self.runTestAttackPerformance(attack))

        for i in range(len(success_rates) - 1):
            self.assertGreaterEqual(success_rates[i + 1], success_rates[i])
        self.assertGreaterEqual(success_rates[-1], 0.95)

    def testBatchGradientDescentUnnormalizedIterationsRelative(self):
        success_rates = []
        epsilon = 0.3

        for iterations in [5, 30]:
            attack = attacks.batch_gradient_descent.BatchGradientDescent()
            attack.max_iterations = iterations
            attack.base_lr = 100
            attack.momentum = 0
            attack.c = 0
            attack.lr_factor = 1.5
            attack.backtrack = False
            attack.normalized = False
            attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
            attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
            attack.norm = attacks.norms.LInfNorm()

            success_rates.append(self.runTestAttackPerformance(attack))

        for i in range(len(success_rates) - 1):
            self.assertGreaterEqual(success_rates[i + 1], success_rates[i])
        self.assertGreaterEqual(success_rates[-1], 0.6)

    def testBatchGradientDescentNormalizedIterationsRelative(self):
        success_rates = []
        epsilon = 0.3

        for iterations in [5, 30]:
            attack = attacks.batch_gradient_descent.BatchGradientDescent()
            attack.max_iterations = iterations
            attack.base_lr = 0.1
            attack.momentum = 0
            attack.c = 0
            attack.lr_factor = 1.5
            attack.backtrack = False
            attack.normalized = True
            attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
            attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
            attack.norm = attacks.norms.LInfNorm()

            success_rates.append(self.runTestAttackPerformance(attack))

        for i in range(len(success_rates) - 1):
            self.assertGreaterEqual(success_rates[i + 1], success_rates[i])
        self.assertGreaterEqual(success_rates[-1], 0.95)

    def testBatchGradientDescentNormalizedBacktrackIterationsRelative(self):
        success_rates = []
        epsilon = 0.3

        for iterations in [5, 30]:
            attack = attacks.batch_gradient_descent.BatchGradientDescent()
            attack.max_iterations = iterations
            attack.base_lr = 0.1
            attack.momentum = 0
            attack.c = 0
            attack.lr_factor = 1.5
            attack.backtrack = True
            attack.normalized = True
            attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
            attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
            attack.norm = attacks.norms.LInfNorm()

            success_rates.append(self.runTestAttackPerformance(attack))

        for i in range(len(success_rates) - 1):
            self.assertGreaterEqual(success_rates[i + 1], success_rates[i])
        self.assertGreaterEqual(success_rates[-1], 0.95)

    def testBatchGradientDescentNormalizedBacktrackF7P(self):
        epsilon = 0.3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 5
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rate = self.runTestAttackPerformance(attack, attempts=5, objective=attacks.objectives.UntargetedF7PObjective())
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchGradientDescentNormalizedBacktrackF7PL2(self):
        epsilon = 3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 5
        attack.base_lr = 1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.L2UniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.L2Projection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.L2Norm()

        success_rate = self.runTestAttackPerformance(attack, attempts=5, objective=attacks.objectives.UntargetedF7PObjective())
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchGradientDescentNormalizedBacktrackF7PL1(self):
        epsilon = 10
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 50
        attack.base_lr = 25
        attack.momentum = 0.9
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.scaled = False
        attack.backtrack = True
        attack.initialization = attacks.initializations.L1UniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.L1Projection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.L1Norm()

        success_rate = self.runTestAttackPerformance(attack, attempts=5, objective=attacks.objectives.UntargetedF7PObjective())
        self.assertGreaterEqual(success_rate, 0.3)
        # Does not really work in normalized mode for some reason!

    def testBatchGradientDescentScaledBacktrackF7PL1(self):
        epsilon = 10
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 50
        attack.base_lr = 5
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = False
        attack.scaled = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.L1UniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.L1Projection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.L1Norm()

        success_rate = self.runTestAttackPerformance(attack, attempts=5, objective=attacks.objectives.UntargetedF7PObjective())
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchGradientDescentNormalizedBacktrackF7PL0(self):
        epsilon = 12
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 100
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.L0UniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.L0Projection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.L0Norm()

        success_rate = self.runTestAttackPerformance(attack, attempts=5, objective=attacks.objectives.UntargetedF7PObjective())
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchGradientDescentDeterministicNormalized(self):
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = True

        self.runTestDeterministicAttack(attack)

    def testBatchGradientDescentDeterministicNormalizedBacktrack(self):
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.1
        attack.normalized = True
        attack.backtrack = True

        self.runTestDeterministicAttack(attack)

    def testBatchSmoothedGradientDescentUnnormalized(self):
        epsilon = 0.3
        attack = attacks.BatchSmoothedGradientDescent()
        attack.population = 5
        attack.variance = 0.01
        attack.max_iterations = 20
        attack.base_lr = 100
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = False  # !
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.6)

    def testBatchSmoothedGradientDescentNormalized(self):
        epsilon = 0.3
        attack = attacks.BatchSmoothedGradientDescent()
        attack.population = 5
        attack.variance = 0.01
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchSmoothedGradientDescentNormalizedBacktrack(self):
        epsilon = 0.3
        attack = attacks.BatchSmoothedGradientDescent()
        attack.population = 5
        attack.variance = 0.01
        attack.max_iterations = 5
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchRandomIterationsRelative(self):
        success_rates = []
        epsilon = 0.3

        for iterations in [5, 1000]:
            attack = attacks.batch_random.BatchRandom()
            attack.max_iterations = iterations
            attack.norm = attacks.norms.LInfNorm()
            attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)

            success_rates.append(self.runTestAttackPerformance(attack))

        for i in range(len(success_rates) - 1):
            self.assertGreaterEqual(success_rates[i + 1], success_rates[i])

    def testBatchZOOUnnormalized(self):
        epsilon = 0.3

        attack = attacks.BatchZOO()
        attack.max_iterations = 1000
        attack.base_lr = 500
        attack.lr_factor = 1
        attack.momentum = 0
        attack.c = 0
        attack.backtrack = False
        attack.normalized = False
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rate = self.runTestAttackPerformance(attack, attempts=2)
        self.assertGreaterEqual(success_rate, 0.4)

    def testBatchZOONormalized(self):
        epsilon = 0.3

        attack = attacks.BatchZOO()
        attack.max_iterations = 1000
        attack.base_lr = 500
        attack.lr_factor = 1
        attack.momentum = 0
        attack.c = 0
        attack.backtrack = False
        attack.normalized = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rate = self.runTestAttackPerformance(attack, attempts=2)
        self.assertGreaterEqual(success_rate, 0.5)

    def testBatchZOONormalizedBacktrack(self):
        epsilon = 0.3

        attack = attacks.BatchZOO()
        attack.max_iterations = 1000
        attack.base_lr = 500
        attack.lr_factor = 1
        attack.momentum = 0
        attack.c = 0
        attack.backtrack = True
        attack.normalized = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rate = self.runTestAttackPerformance(attack, attempts=2)
        self.assertGreaterEqual(success_rate, 0.6)

    def testBatchQueryLimitedUnnormalized(self):
        epsilon = 0.3

        attack = attacks.BatchQueryLimited()
        attack.max_iterations = 50
        attack.base_lr = 10
        attack.lr_factor = 1
        attack.momentum = 0
        attack.c = 0
        attack.population = 50
        attack.variance = 0.3
        attack.backtrack = False
        attack.normalized = False
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rate = self.runTestAttackPerformance(attack, attempts=2)
        self.assertGreaterEqual(success_rate, 0.4)

    def testBatchQueryLimitedNormalized(self):
        epsilon = 0.3

        attack = attacks.BatchQueryLimited()
        attack.max_iterations = 50
        attack.base_lr = 0.05
        attack.lr_factor = 1
        attack.momentum = 0
        attack.c = 0
        attack.population = 50
        attack.variance = 0.1
        attack.backtrack = False
        attack.normalized = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rate = self.runTestAttackPerformance(attack, attempts=2)
        self.assertGreaterEqual(success_rate, 0.5)

    def testBatchQueryLimitedNormalizedBacktrack(self):
        epsilon = 0.3

        attack = attacks.BatchQueryLimited()
        attack.max_iterations = 50
        attack.base_lr = 0.05
        attack.lr_factor = 1.5
        attack.momentum = 0
        attack.c = 0
        attack.population = 50
        attack.variance = 0.1
        attack.backtrack = True
        attack.normalized = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rate = self.runTestAttackPerformance(attack, attempts=2)
        self.assertGreaterEqual(success_rate, 0.6)

    def testBatchSimple(self):
        attack = attacks.BatchSimple()
        attack.max_iterations = 1000
        attack.epsilon = 0.3
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.BoxProjection()])

        # should enforce epsilon constraint without projection!
        self.runTestAttackProjections(attack, attack.epsilon)
        success_rate = self.runTestAttackPerformance(attack, attempts=2)
        self.assertGreaterEqual(success_rate, 0.15)

    def testBatchCube(self):
        epsilon = 0.3
        attack = attacks.BatchCube()
        attack.max_iterations = 1000
        attack.epsilon = epsilon
        attack.probability = 0.1
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])

        self.runTestAttackProjections(attack, attack.epsilon)
        success_rate = self.runTestAttackPerformance(attack, attempts=2)
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchCube2LInf(self):
        epsilon = 0.3
        attack = attacks.BatchCube2()
        attack.max_iterations = 1000
        attack.epsilon = epsilon
        attack.probability = 0.05
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, attack.epsilon)
        success_rate = self.runTestAttackPerformance(attack, attempts=2)
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchCube2L2(self):
        attack = attacks.BatchCube2()
        attack.max_iterations = 1000
        attack.epsilon = 1.5
        attack.probability = 0.05
        attack.norm = attacks.norms.L2Norm()

        self.runTestAttackProjections(attack, attack.epsilon)
        success_rate = self.runTestAttackPerformance(attack, attempts=2)
        self.assertGreaterEqual(success_rate, 0.85)

    def testBatchGeometry(self):
        epsilon = 0.3

        attack = attacks.BatchGeometry()
        attack.database = numpy.transpose(self.testset.images[:1000], (0, 3, 1, 2))
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack, attempts=2)
        self.assertGreaterEqual(success_rate, 0.7)

    def testCornerSearch(self):
        epsilon = 10
        attack = attacks.batch_corner_search.BatchCornerSearch()
        attack.max_iterations = 2
        attack.epsilon = epsilon
        attack.sigma = False

        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.45)

    def testBatchAffine(self):
        N_theta = 6
        min_bound = [-2, -2, -0.25, -0.25, 1.1, -math.pi/8]
        max_bound = [2, 2, 0.25, 0.25, 1.25, math.pi/8]

        optimizer = torch.optim.SGD
        lr = 0.5
        momentum = 0.9

        min_bound = torch.from_numpy(numpy.array(min_bound).reshape(1, N_theta))
        max_bound = torch.from_numpy(numpy.array(max_bound).reshape(1, N_theta))

        if self.cuda:
            min_bound = min_bound.cuda()
            max_bound = max_bound.cuda()

        attack = attacks.batch_affine.BatchAffine(optimizer, lr=lr, momentum=momentum)
        attack.N_theta = N_theta
        attack.max_iterations = 25
        attack.c = 0
        attack.normalized = True
        attack.norm = attacks.norms.LInfNorm()
        attack.initialization = attacks.initializations.UniformInitialization(-0.25, 0.25)
        attack.projection = attacks.projections.BoxProjection(min_bound=min_bound, max_bound=max_bound)

        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.9)

    # test without projection epsilon is left
    # test with c norm is lower
    # test with larger c norm is lower than with smaller c
    # test random attack


class TestAttacksNormalMNISTLeNet(TestAttacksNormalMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'mnist_lenet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.LeNet(10, [1, 28, 28])


class TestAttacksNormalMNISTResNet(TestAttacksNormalMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'mnist_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.ResNet(10, [1, 28, 28], blocks=[1, 1, 1])


class TestAttacksNormalMNISTNormalizedResNet(TestAttacksNormalMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'mnist_normalized_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        model = models.ResNet(10, [1, 28, 28], blocks=[1, 1, 1], whiten=True)

        mean = numpy.zeros(1)
        std = numpy.zeros(1)
        mean[0] = numpy.mean(cls.trainset.images[:, :, :, 0])
        std[0] = numpy.std(cls.trainset.images[:, :, :, 0])

        model.whiten.weight.data = torch.from_numpy(std.astype(numpy.float32))
        model.whiten.bias.data = torch.from_numpy(mean.astype(numpy.float32))

        return model

    def testBatchOptimSGDUnnormalizedZero(self):
        epsilon = 0.3
        optimizer = torch.optim.SGD
        lr = 100
        momentum = 0
        attack = attacks.batch_optim.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 20
        attack.c = 0
        attack.normalized = False
        attack.initialization = None
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.45)

    def testBatchGradientDescentUnnormalizedZero(self):
        epsilon = 0.3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 20
        attack.base_lr = 100
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = False
        attack.initialization = None
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.45)


class TestAttacksNormalMNISTScaledResNet(TestAttacksNormalMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'mnist_scaled_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.ResNet(10, [1, 28, 28], blocks=[1, 1, 1], scale=True)

    def testBatchOptimSGDUnnormalizedZero(self):
        epsilon = 0.3
        optimizer = torch.optim.SGD
        lr = 100
        momentum = 0
        attack = attacks.batch_optim.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 20
        attack.c = 0
        attack.normalized = False
        attack.initialization = None
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.35)

    def testBatchGradientDescentUnnormalizedZero(self):
        epsilon = 0.3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 20
        attack.base_lr = 100
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = False
        attack.initialization = None
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.35)


class TestAttacksNormalSVHNResNet(TestAttacksNormalMNISTMLP):
    @classmethod
    def setDatasets(cls):
        cls.trainset = common.datasets.SVHNTrainSet()
        cls.testset = common.datasets.SVHNTestSet()
        cls.adversarialset = common.datasets.SVHNTestSet(indices=range(100))
        cls.trainloader = torch.utils.data.DataLoader(cls.trainset, batch_size=cls.batch_size, shuffle=True, num_workers=0)
        cls.testloader = torch.utils.data.DataLoader(cls.testset, batch_size=cls.batch_size, shuffle=False, num_workers=0)
        cls.adversarialloader = torch.utils.data.DataLoader(cls.adversarialset, batch_size=cls.batch_size, shuffle=False, num_workers=0)

    @classmethod
    def getModelFile(cls):
        return 'svhn_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.ResNet(10, [3, 32, 32], blocks=[2, 2, 2])


if __name__ == '__main__':
    unittest.main()
