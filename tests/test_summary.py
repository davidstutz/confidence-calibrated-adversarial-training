import unittest
import numpy
import sys
import os

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.summary
import tensorflow
import torch.utils.tensorboard
import datetime


class TestSummary(unittest.TestCase):
    def testSummaryReader(self):
        dt = datetime.datetime.now()
        log_dir = 'logs/%s' % dt.strftime('%d%m%y%H%M%S')
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        original_values = numpy.linspace(0, 1, 10)

        for i in range(original_values.shape[0]):
            writer.add_scalar('value', original_values[i], global_step=i)
        writer.flush()

        reader = common.summary.SummaryTensorboardReader('logs/')
        values, steps = reader.get_scalar(dt.strftime('%d%m%y%H%M%S'), 'value')

        numpy.testing.assert_equal(steps, list(range(values.shape[0])))
        numpy.testing.assert_almost_equal(values, original_values)


if __name__ == '__main__':
    unittest.main()
