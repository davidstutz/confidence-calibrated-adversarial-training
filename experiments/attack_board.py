import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import importlib
import common.experiments
import common.utils
import common.paths


class AttackBoard:
    """
    Start Tensorboard for an attack run.
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
        parser.add_argument('config_target_variable')
        parser.add_argument('config_attack_variable', type=str)
        parser.add_argument('-run', type=str, default='')
        parser.add_argument('-port', type=str, default='6005')

        return parser

    def main(self):
        """
        Main.
        """

        module = importlib.import_module(self.args.config_module)
        assert getattr(module, self.args.config_target_variable, None) is not None
        assert getattr(module, self.args.config_attack_variable, None) is not None
        target_config = getattr(module, self.args.config_target_variable)
        attack_config = getattr(module, self.args.config_attack_variable)
        assert not isinstance(attack_config, list)

        log_dir = common.paths.log_dir('%s/%s' % (target_config.directory, attack_config.directory))
        os.system('python3 -m tensorboard.main --logdir="%s/%s" --host=localhost --port=%s' % (log_dir, self.args.run, self.args.port))


if __name__ == '__main__':
    program = AttackBoard()
    program.main()
