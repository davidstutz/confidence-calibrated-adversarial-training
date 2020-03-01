import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import importlib
import common.experiments
import common.utils
import common.paths
from common.log import Log


class Train:
    """
    Train a model.
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
        # main arguments
        parser.add_argument('interface', type=str)
        parser.add_argument('config_module', type=str)
        parser.add_argument('config_variable', type=str)

        return parser

    def main(self):
        """
        Main.
        """

        module = importlib.import_module(self.args.config_module)
        assert getattr(module, self.args.config_variable, None) is not None, self.args.config_variable
        config = getattr(module, self.args.config_variable)
        interface = common.utils.get_class('common.experiments', self.args.interface)

        program = interface(config)
        program.main()


if __name__ == '__main__':
    program = Train()
    program.main()