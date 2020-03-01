import unittest
import torch
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.state


class TestState(unittest.TestCase):
    def setUp(self):
        self.filepath = 'test.pth.tar'
        if os.path.exists(self.filepath):
            os.unlink(self.filepath)

    def testModels(self):
        model_classes = [
            'LeNet',
            'MLP',
            'ResNet'
        ]

        for model_class in model_classes:
            model_class = common.utils.get_class('models', model_class)
            original_model = model_class(10, [1, 32, 32])
            for parameters in original_model.parameters():
                parameters.data.zero_()

            state = common.state.State(original_model)
            state.save(self.filepath)

            state = common.state.State.load(self.filepath)
            loaded_model = state.model

            self.assertEqual(loaded_model.__class__.__name__, original_model.__class__.__name__)
            self.assertListEqual(loaded_model.resolution, original_model.resolution)

            for parameters in loaded_model.parameters():
                self.assertEqual(torch.sum(parameters).item(), 0)

    def testModelOnly(self):
        original_model = models.LeNet(10, [1, 32, 32])
        for parameters in original_model.parameters():
            parameters.data.zero_()

        state = common.state.State(original_model)
        state.save(self.filepath)

        state = common.state.State.load(self.filepath)
        loaded_model = state.model

        self.assertEqual(loaded_model.__class__.__name__, original_model.__class__.__name__)
        self.assertListEqual(loaded_model.resolution, original_model.resolution)

        for parameters in loaded_model.parameters():
            self.assertEqual(torch.sum(parameters).item(), 0)

    def testModelOptimizer(self):
        original_model = models.LeNet(10, [1, 32, 32])
        original_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
        state = common.state.State(original_model, original_optimizer)
        state.save(self.filepath)

        state = common.state.State.load(self.filepath)
        loaded_model = state.model
        loaded_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.99, momentum=0.1)
        loaded_optimizer.load_state_dict(state.optimizer)

        for param_group in loaded_optimizer.param_groups:
            self.assertEqual(param_group['lr'], 0.01)
            self.assertEqual(param_group['momentum'], 0.9)

    def testModelOptimizerScheduler(self):
        original_model = models.LeNet(10, [1, 32, 32])
        original_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
        original_scheduler = torch.optim.lr_scheduler.StepLR(original_optimizer, step_size=10, gamma=0.9)
        state = common.state.State(original_model, original_optimizer, original_scheduler)
        state.save(self.filepath)

        state = common.state.State.load(self.filepath)
        loaded_model = state.model
        loaded_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.99, momentum=0.1)
        loaded_optimizer.load_state_dict(state.optimizer)
        loaded_scheduler = torch.optim.lr_scheduler.StepLR(original_optimizer, step_size=10, gamma=0.9)
        loaded_scheduler.load_state_dict(state.scheduler)

        self.assertEqual(original_scheduler.step_size, loaded_scheduler.step_size)
        self.assertEqual(original_scheduler.gamma, loaded_scheduler.gamma)

    def testModelOptimizerSchedulerEpoch(self):
        original_model = models.LeNet(10, [1, 32, 32])
        original_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
        original_scheduler = torch.optim.lr_scheduler.StepLR(original_optimizer, step_size=10, gamma=0.9)
        original_epoch = 100
        state = common.state.State(original_model, original_optimizer, original_scheduler, original_epoch)
        state.save(self.filepath)

        state = common.state.State.load(self.filepath)
        loaded_model = state.model
        loaded_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.99, momentum=0.1)
        loaded_optimizer.load_state_dict(state.optimizer)
        loaded_scheduler = torch.optim.lr_scheduler.StepLR(original_optimizer, step_size=10, gamma=0.9)
        loaded_scheduler.load_state_dict(state.scheduler)
        loaded_epoch = state.epoch

        self.assertEqual(original_epoch, loaded_epoch)

    def testLeNet(self):
        resolutions = [
            [1, 2, 2],
            [1, 3, 3],
            [1, 4, 4],
            [1, 5, 5],
            [1, 4, 5],
            [1, 5, 4],
            [1, 27, 32],
            [1, 32, 27],
            [1, 32, 32],
            [3, 32, 32],
        ]
        channels = [1, 2]
        activations = [
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
        ]
        normalizations = [
            True,
            False
        ]

        clamps = [
            True,
            False
        ]

        scales_and_whitens = [
            (False, False),
            (True, False),
            (False, True),
        ]

        classes = 10
        for resolution in resolutions:
            for channel in channels:
                for activation in activations:
                    for normalization in normalizations:
                        for clamp in clamps:
                            for scale_and_whiten in scales_and_whitens:
                                original_model = models.LeNet(classes, resolution, clamp=clamp, scale=scale_and_whiten[0], whiten=scale_and_whiten[1], channels=channel, activation=activation, normalization=normalization)
                                for parameters in original_model.parameters():
                                    parameters.data.zero_()

                                common.state.State.checkpoint(self.filepath, original_model)
                                state = common.state.State.load(self.filepath)
                                loaded_model = state.model

                                for parameters in loaded_model.parameters():
                                    self.assertEqual(torch.sum(parameters).item(), 0)

    def testMLP(self):
        resolutions = [
            [1, 2, 2],
            [1, 3, 3],
            [1, 4, 4],
            [1, 5, 5],
            [1, 4, 5],
            [1, 5, 4],
            [1, 27, 32],
            [1, 32, 27],
            [1, 32, 32],
            [3, 32, 32],
        ]
        units = [
            [10],
            [10, 10, 10, 10],
            [1000],
        ]
        activations = [
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
        ]
        normalizations = [
            None,
            torch.nn.BatchNorm1d
        ]

        clamps = [
            True,
            False
        ]

        scales_and_whitens = [
            (False, False),
            (True, False),
            (False, True),
        ]

        classes = 10
        for resolution in resolutions:
            for unit in units:
                for activation in activations:
                    for normalization in normalizations:
                        for clamp in clamps:
                            for scale_and_whiten in scales_and_whitens:
                                original_model = models.MLP(classes, resolution, clamp=clamp, scale=scale_and_whiten[0], whiten=scale_and_whiten[1], units=unit, activation=activation, normalization=normalization)
                                for parameters in original_model.parameters():
                                    parameters.data.zero_()

                                common.state.State.checkpoint(self.filepath, original_model)
                                state = common.state.State.load(self.filepath)
                                loaded_model = state.model

                                for parameters in loaded_model.parameters():
                                    self.assertEqual(torch.sum(parameters).item(), 0)

    def testResNet(self):
        resolutions = [
            [3, 32, 32],
        ]

        blocks = [
            [3],
            [3, 3, 3, 3],
        ]
        normalizations = [
            None,
            torch.nn.BatchNorm1d
        ]

        clamps = [
            True,
            False
        ]

        scales_and_whitens = [
            (False, False),
            (True, False),
            (False, True),
        ]

        classes = 10
        for resolution in resolutions:
            for block in blocks:
                for normalization in normalizations:
                    for clamp in clamps:
                        for scale_and_whiten in scales_and_whitens:
                            original_model = models.ResNet(classes, resolution, clamp=clamp, scale=scale_and_whiten[0], whiten=scale_and_whiten[1], blocks=block, normalization=normalization)
                            for parameters in original_model.parameters():
                                parameters.data.zero_()

                            common.state.State.checkpoint(self.filepath, original_model)
                            state = common.state.State.load(self.filepath)
                            loaded_model = state.model

                            for parameters in loaded_model.parameters():
                                self.assertEqual(torch.sum(parameters).item(), 0)

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.unlink(self.filepath)


if __name__ == '__main__':
    unittest.main()
