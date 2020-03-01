import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import importlib
import common.experiments
import common.utils
import common.paths
from common.log import log, LogLevel


class TrainBoard:
    """
    Start Tensorboard for a specific training run.
    """

    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        self.args = None
        """ Arguments of program. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('config_module', type=str)
        parser.add_argument('config_variable', type=str)
        parser.add_argument('-port', type=str, default='6005')

        return parser

    def main(self):
        """
        Main.
        """

        module = importlib.import_module(self.args.config_module)
        config = getattr(module, self.args.config_variable)
        assert getattr(module, self.args.config_variable, None) is not None

        log_dir = os.path.join(common.paths.log_dir(config.directory), 'logs')
        if not os.path.exists(log_dir):
            log('Log directory %s does not exist!' % log_dir, LogLevel.ERROR)
            exit()

        log('log directory=%s' % log_dir)
        os.system('python3 -m tensorboard.main --logdir="%s" --host=localhost --port=%s' % (log_dir, self.args.port))


if __name__ == '__main__':
    program = TrainBoard()
    program.main()
