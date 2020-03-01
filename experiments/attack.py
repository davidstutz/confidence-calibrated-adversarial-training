import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import common.experiments
import common.utils
import common.paths
import common.summary
import importlib
import copy
import time
from common.log import log


class Attack:
    """
    Attack a model.
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
        parser.add_argument('config_module', type=str)
        parser.add_argument('config_target_variable', type=str)
        parser.add_argument('config_attack_variable', type=str)
        parser.add_argument('-snapshot', action='store_true', default=False)
        parser.add_argument('-wait', action='store_true', default=False)

        return parser

    def main(self):
        """
        Main.
        """

        module = importlib.import_module(self.args.config_module)
        assert getattr(module, self.args.config_target_variable, None) is not None, self.args.config_target_variable
        assert getattr(module, self.args.config_attack_variable, None) is not None, self.args.config_attack_variable
        target_configs = getattr(module, self.args.config_target_variable)
        if not isinstance(target_configs, list):
            target_configs = [self.args.config_target_variable]
        for target_config in target_configs:
            assert getattr(module, target_config, None) is not None, target_config
        attack_configs = getattr(module, self.args.config_attack_variable)
        if not isinstance(attack_configs, list):
            attack_configs = [self.args.config_attack_variable]
        for attack_config in attack_configs:
            assert getattr(module, attack_config, None) is not None, attack_config

        def no_writer(log_dir, sub_dir=''):
            return common.summary.SummaryWriter(log_dir)

        for j in range(len(target_configs)):
            target_config = getattr(module, target_configs[j])

            if self.args.snapshot and target_config.snapshot is not None:
                wait = True
                epochs = range(target_config.epochs, -1, -target_config.snapshot)

                while True:
                    for i in range(len(attack_configs)):
                        for epoch in epochs:
                            if epoch < target_config.epochs:
                                model_file = common.paths.experiment_file(target_config.directory, 'classifier', common.paths.STATE_EXT + '.%d' % epoch)
                            else:
                                model_file = common.paths.experiment_file(target_config.directory, 'classifier', common.paths.STATE_EXT)
                                if os.path.exists(model_file):
                                    wait = False

                            if os.path.exists(model_file):
                                attack_config = getattr(module, attack_configs[i])
                                log(attack_config.directory)

                                snapshot_attack_config = copy.deepcopy(attack_config)
                                snapshot_attack_config.get_writer = no_writer
                                if epoch < target_config.epochs:
                                    snapshot_attack_config.snapshot = epoch

                                program = common.experiments.AttackInterface(target_config, snapshot_attack_config)
                                program.main()
                                break;

                    if not self.args.wait:
                        break
                    if not wait:
                        break

                    #log('%s' % model_file)
                    time.sleep(0.1)

            else:
                for i in range(len(attack_configs)):
                    attack_config = getattr(module, attack_configs[i])
                    program = common.experiments.AttackInterface(target_config, attack_config)
                    program.main()


if __name__ == '__main__':
    program = Attack()
    program.main()