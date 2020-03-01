"""
Utilities for the evaluation in Jupyer notebooks.
"""

import os
import common.utils
import common.test
import common.eval
import common.paths
import common.state
import common.utils
import importlib
import numpy
import terminaltables
import datetime


def add_attacks(config_attack_variables, attacks, identifier):
    """
    Add given attacks to given list of attack variables. Allows to keep an overview.

    :param config_attack_variables: list of attack config names
    :type config_attack_variables: [str]
    :param attacks: attack config names to add
    :type attacks: [str]
    :param identifier: which attacks were added:
    :type identifier: str
    :return: indices where set of attacks where added (start and end)
    :rtype: (int, int)
    """

    start = len(config_attack_variables)
    config_attack_variables += attacks
    end = len(config_attack_variables)
    print('%s: %d-%d' % (identifier, start, end))
    return list(range(start, end))


def module(config_module, config_training_variables, config_attack_variables):
    """
    Load module and training as well as attack configuration.

    :param config_module: config module such as config.mnist
    :type config_module: str
    :param config_training_variables: list of training variables
    :type config_training_variables: [str]
    :param config_attack_variables: list of attack variables
    :type config_attack_variables: [str]
    :return: config, training configurations, attack configurations
    :rtype: module, [common.experiments.NormalTrainingConfig or similar], [common.experiments.AttackConfig]
    """

    config = importlib.import_module('experiments.' + config_module)

    if isinstance(config_training_variables, list):
        config_training_variables = config_training_variables
    else:
        config_training_variables = getattr(config, config_training_variables)
        if not isinstance(config_training_variables, list):
            config_training_variables = [config_training_variables]
    training_configs = [getattr(config, config_training_variable) for config_training_variable in config_training_variables]
    for training_config in training_configs:
        training_config.validate()

    if isinstance(config_attack_variables, list):
        config_attack_variables = config_attack_variables
    else:
        config_attack_variables = getattr(config, config_attack_variables)
        if not isinstance(config_attack_variables, list):
            config_attack_variables = [config_attack_variables]
    attack_configs = [getattr(config, config_attack_variable) for config_attack_variable in config_attack_variables]
    for attack_config in attack_configs:
        attack_config.validate()

    return config, training_configs, attack_configs


def load(training_configs, config_training_names, attack_configs):
    """
    Load model files and perturbation (i.e., adversarial example) files.

    :param training_configs: training configurations obtained from module()
    :type training_configs: [experiments.NormalTrainingConfig or similar]
    :param config_training_names: training configuration variables
    :type config_training_names: [str]
    :param attack_configs: attack configurations obtained from module()
    :type attack_configs: [experiments.AttackConfig]
    :return: model files, model epochs, perturbation files, perturbation epochs, config training variables, config training names
    :rtype: [str], [int], [str], [int], [str], [str]
    """

    model_files = []
    model_epochs = []
    perturbations_files = []
    perturbations_epochs = []
    new_config_training_variables = []
    new_config_training_names = []

    for i in range(len(training_configs)):
        training_config = training_configs[i]
        model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)

        if os.path.exists(model_file):
            model_files.append(model_file)
            model_epochs.append(training_config.epochs)
            new_config_training_variables.append(training_config.directory)
            new_config_training_names.append(config_training_names[i])
        elif training_config.snapshot is not None:
            for epoch in range(training_config.epochs, -1, -training_config.snapshot):
                model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT + '.%d' % epoch)
                if os.path.exists(model_file):
                    model_files.append(model_file)
                    model_epochs.append(epoch)
                    new_config_training_variables.append(training_config.directory)
                    new_config_training_names.append(config_training_names[i])
                    break;

        perturbations_files_ = []
        perturbations_epochs_ = []

        if len(new_config_training_variables) == 0 or new_config_training_variables[-1] != training_config.directory:
            print('Ignored: %s' % training_config.directory)
        else:
            for attack_config in attack_configs:
                perturbations_file = common.paths.experiment_file('%s/%s' % (training_config.directory, attack_config.directory), 'perturbations',
                                                                  common.paths.HDF5_EXT)
                perturbations_epoch = training_config.epochs

                if not os.path.exists(perturbations_file):
                    for epoch in range(training_config.epochs, -1, -training_config.snapshot):
                        perturbations_file_ = common.paths.experiment_file('%s/%s_%d' % (training_config.directory, attack_config.directory, epoch),
                                                                           'perturbations', common.paths.HDF5_EXT)
                        if os.path.exists(perturbations_file_):
                            perturbations_file = perturbations_file_
                            perturbations_epoch = epoch
                            break;

                perturbations_files_.append(perturbations_file)
                perturbations_epochs_.append(perturbations_epoch)

                # if not os.path.exists(perturbations_file):
                #    display(Markdown('**<font color="red">Ignored: %s</font>**' % perturbations_file))

            perturbations_files.append(perturbations_files_)
            perturbations_epochs.append(perturbations_epochs_)

    # some perturbations files might no exist!
    for model_file in model_files:
        assert os.path.exists(model_file), model_file
    assert len(model_epochs) == len(model_files), (len(model_epochs), len(model_files))
    if len(attack_configs) > 0:
        assert len(model_files) == len(perturbations_files), (len(model_files), len(perturbations_files))
        assert len(perturbations_files) == len(perturbations_epochs), (len(perturbations_files), len(perturbations_epochs))

    return model_files, model_epochs, perturbations_files, perturbations_epochs, new_config_training_variables, new_config_training_names


def epoch_table(model_epochs, perturbations_files, perturbations_epochs, config_training_names, attack_configs):
    """
    Create markdown table summarizing epochs of models and adversarial examples.

    :param model_epochs: model epochs as from load()
    :type model_epochs: [int]
    :param perturbations_files: perturbation files as from load()
    :type perturbations_files: [str]
    :param perturbations_epochs: perturbation epochs as from load()
    :type perturbations_epochs: [int]
    :param config_training_names: configuration training names as from load()
    :type config_training_names: [str]
    :param attack_configs: attack configurations as from module()
    :type attack_configs: [experiments.AttackConfig]
    :return: markdown table
    :rtype: str
    """

    table_data = [['Model\Attack', 'Training'] + list(range(len(attack_configs)))]
    for i in range(len(perturbations_files)):
        table_data_ = [str(config_training_names[i]), model_epochs[i]]
        for j in range(len(perturbations_files[i])):
            if not os.path.exists(perturbations_files[i][j]):
                table_data_.append('--')
            else:
                table_data_.append(str(perturbations_epochs[i][j]))

        table_data.append(table_data_)
    table = terminaltables.GithubFlavoredMarkdownTable(table_data)
    return table.table


def load_model(model_file, cuda=True):
    """
    Load a model.

    :param model_file: model file
    :type model_file: str
    :param cuda: use on GPU
    :type cuda: bool
    :return: model
    :rtype: torch.nn.Module
    """

    assert os.path.exists(model_file)

    state = common.state.State.load(model_file)
    model = state.model

    if cuda:
        model = model.cuda()

    model.eval()
    return model, state.epoch


def compute_clean_probabilities(model_files, testloader, cuda=True):
    """
    Compute probabilities of clean model on test loader.

    :param model_files: model files
    :type model_files: [str]
    :param testloader: test loader
    :type testloader: torch.utils.data.DataLoader
    :param cuda: use GPU
    :type cuda: bool
    :return: probabilities for each model
    :rtype: [numpy.ndarray]
    """

    clean_probabilities = []
    for i in range(len(model_files)):
        model, _ = load_model(model_files[i])
        clean_probabilities.append(common.test.test(model, testloader, cuda))
    assert len(clean_probabilities) == len(model_files)
    return clean_probabilities


def compute_clean_evaluations(config, clean_probabilities):
    """
    Clean evaluation on probabilities.

    :param config: configuration as from module()
    :type config: module
    :param clean_probabilities: list of clean probabilities for each modelas from compute_clean_probabilities()
    :type clean_probabilities: [numpy.ndarray]
    :return: clean evaluations
    :rtype: [common.eval.CleanEvaluation]
    """

    clean_evaluations = []
    for i in range(len(clean_probabilities)):
        clean_evaluations.append(common.eval.CleanEvaluation(clean_probabilities[i], config.testset.labels))
    return clean_evaluations


def load_adversarial_probabilities(perturbations_files):
    """
    Load adversarial probabilities from perturbation files.
    
    :param perturbations_files: perturbation files as from load()
    :type perturbations_files: [str]
    :return: list of proabilities for each model and for each attack
    :rtype: [[numpy.ndarray]]
    """

    adversarial_probabilities = []
    for i in range(len(perturbations_files)):
        adversarial_probabilities_ = []
        for j in range(len(perturbations_files[i])):
            if os.path.exists(perturbations_files[i][j]):
                adversarial_probabilities_.append(common.utils.read_hdf5(perturbations_files[i][j], key='probabilities'))
            else:
                adversarial_probabilities_.append(None)
        adversarial_probabilities.append(adversarial_probabilities_)
    return adversarial_probabilities


def compute_adversarial_evaluations(config, perturbations_files, adversarial_probabilities, clean_probabilities, config_training_variable, config_attack_variable, groups, evaluator=common.eval.AdversarialEvaluation):
    """
    Compute evaluation of perturbations/adversarial examples.

    :param config: config module as from module()
    :type config: module
    :param perturbations_files: perturbation files
    :type perturbations_files: [str]
    :param adversarial_probabilities: probabilities on perturbations for each model and each attack as from load_adversarial_perturbations()
    :type adversarial_probabilities: [[numpy.ndarray]]
    :param clean_probabilities: probabilities on clean examples for each model
    :type clean_probabilities: [numpy.ndarray]
    :param config_training_variable: config training variables
    :type config_training_variable: [str]
    :param config_attack_variable: config attack variables
    :type config_attack_variable: [str]
    :param groups: attack groups
    :type groups: [[(int, int)]]
    :param evaluator: evaluation clas to use
    :type evaluator: common.eval.AdversarialEvaluation or similar
    :return: adversarial evaluations for each model and each group
    :rtype: [[common.eval.AdversarialEvaluation nor similar]]
    """

    samples = 1000
    individual_adversarial_evaluations = []

    for i in range(len(perturbations_files)):
        assert len(adversarial_probabilities[i]) > 0

        individual_adversarial_evaluations_ = []
        for indices in groups:
            adversarial_probabilities_igroup = None
            errors_igroup = None

            print('group')
            for j in indices:
                print('    ', i, len(adversarial_probabilities), config_training_variable[i], j, len(adversarial_probabilities[i]), config_attack_variable[j])
                if adversarial_probabilities[i][j] is not None:
                    adversarial_probabilities_ij = numpy.copy(adversarial_probabilities[i][j][:, :samples])
                    # print(config.testset.labels[:10],
                    #      numpy.argmax(adversarial_probabilities_ij[0, :10, :], axis=1),
                    #      numpy.max(adversarial_probabilities_ij[0, :10, :], axis=1)
                    #     )
                    # print(adversarial_probabilities_ij[0, :10])
                    adversarial_probabilities_ij[
                        :,
                        numpy.arange(adversarial_probabilities_ij.shape[1]),
                        config.testset.labels[:adversarial_probabilities_ij.shape[1]],
                    ] = 0
                    # print(adversarial_probabilities_ij[0, :10])
                    # print(adversarial_probabilities_ij.shape)
                    assert len(adversarial_probabilities_ij.shape) == 3
                    errors_ij = -numpy.max(adversarial_probabilities_ij, axis=2)
                    # print(errors_ij.shape)

                    adversarial_probabilities_igroup = common.numpy.concatenate(adversarial_probabilities_igroup, adversarial_probabilities[i][j][:, :samples])
                    errors_igroup = common.numpy.concatenate(errors_igroup, errors_ij, axis=0)

                    evaluation_ij = evaluator(clean_probabilities[i],
                                                                      adversarial_probabilities[i][j][:, :samples],
                                                                      config.testset.labels,
                                                                      validation=0.1,
                                                                      errors=errors_ij,
                                                                      include_misclassifications=False,
                                                                      )
                    print('        ', evaluation_ij.fpr_at_99tpr())

            # print(errors_igroup.shape)
            if adversarial_probabilities_igroup is not None:
                evaluation_igroup = evaluator(clean_probabilities[i],
                                                                      adversarial_probabilities_igroup,
                                                                      config.testset.labels,
                                                                      validation=0.1,
                                                                      errors=errors_igroup,
                                                                      include_misclassifications=False,
                                                                      )
                individual_adversarial_evaluations_.append(evaluation_igroup)
                print(evaluation_igroup.fpr_at_99tpr())
            else:
                individual_adversarial_evaluations_.append(None)
        individual_adversarial_evaluations.append(individual_adversarial_evaluations_)
    for i in range(len(individual_adversarial_evaluations)):
        assert len(individual_adversarial_evaluations[i]) == len(groups)
        # assert len(individual_adversarial_evaluations[i])%2 == 0
    return individual_adversarial_evaluations


def main_markdown_table(individual_adversarial_evaluations, config_training_names, config_attack_names, tpr):
    """
    Generate the main results table.
    """

    table_data = [
        ['', 'Attack', '', 'Model', 'TE@0', 'RTE@0', 'ROC AUC', 'FPR@%s%%TPR' % tpr, 'TE@%s%%TPR' % tpr, 'RTE@%s%%TPR' % tpr, '\\tau@%s%%TPR' % tpr, 'Test TPR@%s%%TPR' % tpr,
         'Val TPR@%s%%TPR' % tpr]]  # heading
    for j in range(len(individual_adversarial_evaluations[0])):
        for i in range(len(individual_adversarial_evaluations)):
            table_data_ = [
                j,
                config_attack_names[j],
                i,
                config_training_names[i],
            ]

            if individual_adversarial_evaluations[i][j] is None:
                table_data_ += [
                    '--',
                    '--',
                    '--',
                    '--',
                    '--',
                    '--',
                    '--',
                    '--',
                    '--',
                ]
            else:
                table_data_ += [
                    '%.2f' % (round(individual_adversarial_evaluations[i][j].test_error(), 3) * 100),
                    '%.2f' % (round(individual_adversarial_evaluations[i][j].robust_test_error(), 3) * 100),
                    '%.2f' % (round(individual_adversarial_evaluations[i][j].receiver_operating_characteristic_auc(), 2)),
                    '%.2f' % (round(getattr(individual_adversarial_evaluations[i][j], 'fpr_at_%stpr' % tpr)(), 3) * 100),
                    '%.2f' % (round(getattr(individual_adversarial_evaluations[i][j], 'test_error_at_%stpr' % tpr)(), 3) * 100),
                    '%.2f (%.2f)' % (round(getattr(individual_adversarial_evaluations[i][j], 'robust_test_error_at_%stpr' % tpr)(), 3) * 100,
                                     round(getattr(individual_adversarial_evaluations[i][j], 'alternative_robust_test_error_at_%stpr' % tpr)(), 3) * 100),
                    '%.2f' % (round(getattr(individual_adversarial_evaluations[i][j], 'confidence_at_%stpr' % tpr)(), 2)),
                    '%.2f' % (round(getattr(individual_adversarial_evaluations[i][j], 'tpr_at_%stpr' % tpr)(), 3)),
                    '%.3f' % (round(getattr(individual_adversarial_evaluations[i][j], 'validation_tpr_at_%stpr' % tpr)(), 3)),
                ]

            table_data.append(table_data_)
    table = terminaltables.GithubFlavoredMarkdownTable(table_data)
    return table.table


def corrupted_markdown_table(ood_evaluations, config_training_names, ood_names, tpr):
    """
    Generate corrupted markdown table.
    """

    table_data = [['', '', 'Corruption', 'Model', 'Clean TE', 'Corr TE', 'ROC AUC', 'FPR@%s%%TPR' % tpr, 'TNR@%s%%TPR' % tpr, 'Corr TE@%s%%TPR' % tpr,
                   'tau@%s%%TPR' % tpr, 'Test TPR@%s%%TPR' % tpr, 'Val TPR@%s%%TPR' % tpr]]  # heading
    values = []
    for j in range(len(ood_evaluations[0])):
        values_ = []
        for i in range(len(ood_evaluations)):
            assert j < len(ood_names), (j, len(ood_names), len(ood_evaluations[i]))
            table_data_ = [
                i,
                j,
                config_training_names[i],
                ood_names[j],
            ]

            table_values = [
                round(ood_evaluations[i][j].test_error() * 100, 2),
                round(ood_evaluations[i][j].corrupted_test_error() * 100, 2),
                round(ood_evaluations[i][j].receiver_operating_characteristic_auc(), 2),
                round(getattr(ood_evaluations[i][j], 'fpr_at_%stpr' % tpr)(), 3) * 100,
                round(getattr(ood_evaluations[i][j], 'tnr_at_%stpr' % tpr)(), 3) * 100,
                round(getattr(ood_evaluations[i][j], 'corrupted_test_error_at_%stpr' % tpr)() * 100, 2),
                round(getattr(ood_evaluations[i][j], 'confidence_at_%stpr' % tpr)(), 2),
                round(getattr(ood_evaluations[i][j], 'tpr_at_%stpr' % tpr)(), 3) * 100,
                round(getattr(ood_evaluations[i][j], 'validation_tpr_at_%stpr' % tpr)(), 3) * 100,
            ]

            table_data_ += [
                '%.2f' % table_values[0],
                '%.2f' % table_values[1],
                '%.2f' % table_values[2],
                '%.2f' % table_values[3],
                '%.2f' % table_values[4],
                '%.2f' % table_values[5],
                '%.2f' % table_values[6],
                '%.2f' % table_values[7],
                '%.2f' % table_values[8],
            ]
            values_.append(table_values)
            table_data.append(table_data_)
        values.append(values_)

    for i in range(len(ood_evaluations)):
        table_data_ = [
            i
            -1,
            config_training_names[i],
            'mean',
        ]

        table_data_ += [
            '%.2f' % numpy.mean([values[j][i][0] for j in range(len(ood_evaluations[0]))]),
            '%.2f' % numpy.mean([values[j][i][1] for j in range(len(ood_evaluations[0]))]),
            '%.2f' % numpy.mean([values[j][i][2] for j in range(len(ood_evaluations[0]))]),
            '%.2f' % numpy.mean([values[j][i][3] for j in range(len(ood_evaluations[0]))]),
            '%.2f' % numpy.mean([values[j][i][4] for j in range(len(ood_evaluations[0]))]),
            '%.2f' % numpy.mean([values[j][i][5] for j in range(len(ood_evaluations[0]))]),
            '%.2f' % numpy.mean([values[j][i][6] for j in range(len(ood_evaluations[0]))]),
            '%.2f' % numpy.mean([values[j][i][7] for j in range(len(ood_evaluations[0]))]),
            '%.2f' % numpy.mean([values[j][i][8] for j in range(len(ood_evaluations[0]))]),
        ]
        table_data.append(table_data_)

    table = terminaltables.GithubFlavoredMarkdownTable(table_data)
    return table.table


def distal_markdown_table(individual_adversarial_evaluations, config_training_names, config_attack_names, tpr):
    """
    Generate distal markdown table.
    """

    table_data = [
        ['', 'Attack', '', 'Model', 'ROC AUC', 'FPR@%s%%TPR' % tpr, '\\tau@%s%%TPR' % tpr, 'Test TPR@%s%%TPR' % tpr,
         'Val TPR@%s%%TPR' % tpr]]  # heading
    for j in range(len(individual_adversarial_evaluations[0])):
        for i in range(len(individual_adversarial_evaluations)):
            table_data_ = [
                j,
                config_attack_names[j],
                i,
                config_training_names[i],
            ]

            if individual_adversarial_evaluations[i][j] is None:
                table_data_ += [
                    '--',
                    '--',
                    '--',
                    '--',
                    '--',
                ]
            else:
                table_data_ += [
                    '%.2f' % (round(individual_adversarial_evaluations[i][j].receiver_operating_characteristic_auc(), 2)),
                    '%.2f' % (round(getattr(individual_adversarial_evaluations[i][j], 'fpr_at_%stpr' % tpr)(), 3) * 100),
                    '%.2f' % (round(getattr(individual_adversarial_evaluations[i][j], 'confidence_at_%stpr' % tpr)(), 2)),
                    '%.2f' % (round(getattr(individual_adversarial_evaluations[i][j], 'tpr_at_%stpr' % tpr)(), 3)),
                    '%.3f' % (round(getattr(individual_adversarial_evaluations[i][j], 'validation_tpr_at_%stpr' % tpr)(), 3)),
                ]

            table_data.append(table_data_)
    table = terminaltables.GithubFlavoredMarkdownTable(table_data)
    return table.table


def save_latex(filepath, latex):
    """
    Save latex file.

    :param filepath: path to file
    :rtype filepath: str
    :param latex: latex code
    :type latex: str
    """

    with open(filepath, 'w') as f:
        f.write(latex)


def main_latex_table(config, config_training_names, config_attack_names, individual_adversarial_evaluations, tpr):
    """
    Main LaTeX table.
    """

    dataset = config.__name__.split('.')[-1].lower()

    latex = """%% %s
\\begin{tabularx}{1\textwidth}{|X|%s}
\\hline
\\textbf{%s} &
\\multicolumn{%d}{c|}{\\textbf{\\RTE} in \\%% for $\\tau$@$%s\\%%$TPR}\\\\
\\hline
""" % (
        str(datetime.datetime.now()),
        '|'.join(['c' for j in range(len(individual_adversarial_evaluations[0]))]) + '|',
        dataset.upper(),
        len(individual_adversarial_evaluations[0]),
        tpr,
    )

    for j in range(len(individual_adversarial_evaluations[0])):
        latex += '& %s\n' % config_attack_names[j]
    latex += '\\\\\\hline\n'

    for j in range(len(individual_adversarial_evaluations[0])):
        if j == 0:
            latex += '& seen\n'
        else:
            latex += '& unseen\n'
    latex += '\\\\\\hline\n'

    for i in range(len(individual_adversarial_evaluations)):
        latex += config_training_names[i] + ' '
        for j in range(len(individual_adversarial_evaluations[i])):
            if individual_adversarial_evaluations[i][j] is not None:
                latex += '& %.1f %% rte@%s\n' % (
                    round(getattr(individual_adversarial_evaluations[i][j], 'robust_test_error_at_%stpr' % tpr)(), 3) * 100,
                    tpr
                )
            else:
                latex += '& -- '
        latex += '\\\\\n\\hline\n'

    latex += '\\end{tabularx}\n'

    filepath = os.path.join(dataset, 'main_%stpr.tex' % tpr)
    common.utils.makedir(os.path.dirname(filepath))
    save_latex(filepath, latex)

    print(latex)


def main2_latex_table(config, config_training_names, config_attack_names, individual_adversarial_evaluations, tpr):
    """
    Main LaTeX table.
    """

    dataset = config.__name__.split('.')[-1].lower()

    latex = """%% %s
\\begin{tabularx}{1\\textwidth}{|X|%s}
\\hline
\\textbf{%s} &
\\multicolumn{%d}{c|}{\\textbf{\\RTE} in \\%% for $\\tau$@$%s\\%%$TPR}\\\\
\\hline
""" % (
        str(datetime.datetime.now()),
        '|'.join(['c' for j in range(2 * len(individual_adversarial_evaluations[0]))]) + '|',
        dataset.upper(),
        len(individual_adversarial_evaluations[0]),
        tpr,
    )

    for j in range(len(individual_adversarial_evaluations[0])):
        latex += '& \\multicolumn{2}{c|}{%s}\n' % config_attack_names[j]
    latex += '\\\\\\hline\n'

    for j in range(len(individual_adversarial_evaluations[0])):
        if j == 0:
            latex += '& \\multicolumn{2}{c|}{seen}\n'
        else:
            latex += '& \\multicolumn{2}{c|}{unseen}\n'
    latex += '\\\\\\hline\n'

    for j in range(len(individual_adversarial_evaluations[0])):
        latex += '& FPR & \\RTE\n'
    latex += '\\\\\\hline\\hline\n'

    for i in range(len(individual_adversarial_evaluations)):
        latex += config_training_names[i] + ' '
        for j in range(len(individual_adversarial_evaluations[i])):
            if individual_adversarial_evaluations[i][j] is not None:
                latex += '& %.1f & %.1f %% fpr@%s, rte@%s\n' % (
                    round(getattr(individual_adversarial_evaluations[i][j], 'fpr_at_%stpr' % tpr)(), 3) * 100,
                    round(getattr(individual_adversarial_evaluations[i][j], 'robust_test_error_at_%stpr' % tpr)(), 3) * 100,
                    tpr,
                    tpr
                )
            else:
                latex += '& -- '
        latex += '\\\\\n'

    latex += '\\end{tabularx}\n'

    filepath = os.path.join(dataset, 'main2_%stpr.tex' % tpr)
    common.utils.makedir(os.path.dirname(filepath))
    save_latex(filepath, latex)

    print(latex)


def supp_latex_table(config, config_training_names, config_attack_names, individual_adversarial_evaluations, tpr):
    """
    Supplementary LaTeX table.
    """

    dataset = config.__name__.split('.')[-1].lower()

    latex = '%% %s\n' % str(datetime.datetime.now())
    latex += '\\begin{tabularx}{1\\textwidth}{| r | X ||c|c|c|c|c||c|c|}\n'
    latex += '\\hline\n'
    latex += '\\multicolumn{9}{|c|}{\\textbf{%s:} Supplementary Results for Adversarial Examples}\\\\\n' % dataset.upper()
    latex += '\\hline\n'
    latex += '&& \\multicolumn{5}{c||}{\\begin{tabular}{c}\\textbf{Detection Setting}\\\\$\\tau$@$%s\%%$TPR\\end{tabular}} & \\multicolumn{2}{c|}{\\begin{tabular}{c}\\textbf{Standard Setting}\\\\$\\tau{=}0$\\end{tabular}}\\\\\n' % tpr
    latex += '\\hline\n'
    latex += 'Attack & Training & \\begin{tabular}{c}ROC\\\\AUC\\end{tabular} & \\begin{tabular}{c}FPR\\\\ in \\%\\end{tabular} & \\begin{tabular}{c}\\TE\\\\ in \\%\\end{tabular} & \\begin{tabular}{c}\\RTE\\\\ in \\%\\end{tabular} & $\\tau$ & \\begin{tabular}{c}\\TE\\\\ in \\%\\end{tabular} & \\begin{tabular}{c}\\RTE\\\\ in \\%\\end{tabular}\\\\\n'
    latex += '\\hline\n'
    latex += '\\hline\n'
    for j in range(len(individual_adversarial_evaluations[0])):
        latex += '\multirow{%d}{*}{%s} ' % (len(individual_adversarial_evaluations), config_attack_names[j])
        for i in range(len(individual_adversarial_evaluations)):
            if individual_adversarial_evaluations[i][j] is not None:
                latex += '& %s & %.2f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f\\\\\n' % (
                    config_training_names[i],
                    round(individual_adversarial_evaluations[i][j].receiver_operating_characteristic_auc(), 2),
                    round(getattr(individual_adversarial_evaluations[i][j], 'fpr_at_%stpr' % tpr)(), 3) * 100,
                    round(getattr(individual_adversarial_evaluations[i][j], 'test_error_at_%stpr' % tpr)(), 3)*100,
                    round(getattr(individual_adversarial_evaluations[i][j], 'alternative_robust_test_error_at_%stpr' % tpr)(), 3)*100,
                    round(getattr(individual_adversarial_evaluations[i][j], 'confidence_at_%stpr' % tpr)(), 2),
                    round(individual_adversarial_evaluations[i][j].test_error(), 3)*100,
                    round(individual_adversarial_evaluations[i][j].robust_test_error(), 3)*100,
                )
            else:
                latex += '& %s & -- & -- & -- & -- & -- & --\\\\\n' % (
                    config_training_names[i],
                )
        latex += '\\hline\n'
    latex += '\\end{tabularx}\n'

    filepath = os.path.join(dataset, 'supp_%stpr.tex' % tpr)
    common.utils.makedir(os.path.dirname(filepath))
    save_latex(filepath, latex)

    print(latex)


def corrupted_main_latex_table(config, config_training_names, ood_evaluations, tpr):
    """
    Corrupted main LaTeX table.
    """

    dataset = config.__name__.split('.')[-1].lower()

    latex = """%% %s
\\begin{tabular}{|c|}
\\hline
\\\\
\\hline
\\begin{tabular}{@{}c@{}}corr.\\\\\\vphantom{t}\\end{tabular}\\\\
\\hline
unseen\\\\
\\hline
\\TE\\
\\hline\\hline
    """ % str(datetime.datetime.now())

    values = []
    for i in range(len(ood_evaluations)):
        values_ = []
        for j in range(len(ood_evaluations[0])):
            table_values = [
                round(ood_evaluations[i][j].receiver_operating_characteristic_auc(), 2),
                round(getattr(ood_evaluations[i][j], 'fpr_at_%stpr' % tpr)(), 3) * 100,
                round(getattr(ood_evaluations[i][j], 'tnr_at_%stpr' % tpr)(), 3) * 100,
                round(getattr(ood_evaluations[i][j], 'corrupted_test_error_at_%stpr' % tpr)() * 100, 2),
                round(getattr(ood_evaluations[i][j], 'confidence_at_%stpr' % tpr)(), 2),
                round(ood_evaluations[i][j].corrupted_test_error() * 100, 2),
            ]
            values_.append(table_values)
        values.append(values_)

    for i in range(len(ood_evaluations)):
        err = numpy.mean([values[i][j][3] for j in range(len(ood_evaluations[i]))])
        latex += '%.1f\\\\ %% %s %s\n' % (err, config_training_names[i], tpr)
    latex += '\\hline\n'
    latex += '\\end{tabular}\n'

    filepath = os.path.join(dataset, 'corrupted_main_%stpr.tex' % tpr)
    common.utils.makedir(os.path.dirname(filepath))
    save_latex(filepath, latex)

    print(latex)


def corrupted_supp_latex_table(config, config_training_names, ood_names, ood_evaluations, tpr):
    """
    Corrupted supplementary table.
    """

    dataset = config.__name__.split('.')[-1].lower()

    latex = '%% %s\n' % str(datetime.datetime.now())
    latex += '\\begin{tabularx}{1\\textwidth}{| r | X ||c|c|c|c|c||c|}\n'
    latex += '\\hline\n'
    latex += '\\multicolumn{8}{|c|}{\\textbf{%s:} Supplementary Results for Corruption}\\\\\n' % dataset.upper()
    latex += '\\hline\n'
    latex += '&& \\multicolumn{5}{c||}{\\begin{tabular}{c}\\textbf{Detection Setting}\\\\$\\tau$@$%s\%%$TPR\\end{tabular}} & \\multicolumn{1}{c|}{\\begin{tabular}{c}\\textbf{Standard}\\\\\\textbf{Setting}\\\\$\\tau{=}0$\\end{tabular}}\\\\\n' % tpr
    latex += '\\hline\n'
    latex += 'Corruption & Training & \\begin{tabular}{c}ROC\\\\AUC\\end{tabular} & \\begin{tabular}{c}FPR\\\\ in \\%\\end{tabular} & \\begin{tabular}{c}TNR\\\\ in \\%\\end{tabular} & \\begin{tabular}{c}\\TE\\\\ in \\%\\end{tabular} & $\\tau$ & \\begin{tabular}{c}\\TE\\\\ in \\%\\end{tabular}\\\\\n'
    latex += '\\hline\n'
    latex += '\\hline\n'

    values = []
    for i in range(len(ood_evaluations)):
        values_ = []
        for j in range(len(ood_evaluations[0])):
            table_values = [
                round(ood_evaluations[i][j].receiver_operating_characteristic_auc(), 2),
                round(getattr(ood_evaluations[i][j], 'fpr_at_%stpr' % tpr)(), 3) * 100,
                round(getattr(ood_evaluations[i][j], 'tnr_at_%stpr' % tpr)(), 3) * 100,
                round(getattr(ood_evaluations[i][j], 'corrupted_test_error_at_%stpr' % tpr)() * 100, 2),
                round(getattr(ood_evaluations[i][j], 'confidence_at_%stpr' % tpr)(), 2),
                round(ood_evaluations[i][j].corrupted_test_error() * 100, 2),
            ]
            values_.append(table_values)
        values.append(values_)

    j = 0
    for i in range(len(ood_evaluations)):
        if i == 0:
            latex += '\\multirow{%d}{*}{\\texttt{%s}} ' % (len(ood_evaluations), ood_names[j].replace('_', '\\_'))
        latex += '& %s & %.2f & %.1f & %.1f & %.1f & %.1f & %.1f\\\\\n' % (
            config_training_names[i],
            values[i][j][0],
            values[i][j][1],
            values[i][j][2],
            values[i][j][3],
            values[i][j][4],
            values[i][j][5],
        )
    latex += '\\hline\n'

    for i in range(len(ood_evaluations)):
        if i == 0:
            latex += '\\multirow{%d}{*}{\\texttt{%s}} ' % (len(ood_evaluations), 'mean')
        latex += '& %s & %.2f & %.1f & %.1f & %.1f & %.1f & %.1f\\\\\n' % (
            config_training_names[j],
            numpy.mean([values[i][j][0] for j in range(len(ood_evaluations[0]))]),
            numpy.mean([values[i][j][1] for j in range(len(ood_evaluations[0]))]),
            numpy.mean([values[i][j][2] for j in range(len(ood_evaluations[0]))]),
            numpy.mean([values[i][j][3] for j in range(len(ood_evaluations[0]))]),
            numpy.mean([values[i][j][4] for j in range(len(ood_evaluations[0]))]),
            numpy.mean([values[i][j][5] for j in range(len(ood_evaluations[0]))]),
        )
    latex += '\\hline\n'

    for j in range(1, len(ood_evaluations[0])):
        for i in range(len(ood_evaluations)):
            if i == 0:
                latex += '\\multirow{%d}{*}{\\texttt{%s}} ' % (len(ood_evaluations), ood_names[j].replace('_', '\\_'))
            latex += '& %s & %.2f & %.1f & %.1f & %.1f & %.1f & %.1f\\\\\n' % (
                config_training_names[i],
                values[i][j][0],
                values[i][j][1],
                values[i][j][2],
                values[i][j][3],
                values[i][j][4],
                values[i][j][5],
            )
        latex += '\\hline\n'

    latex += '\\hline\n'
    latex += '\\end{tabularx}\n'

    filepath = os.path.join(dataset, 'corrupted_supp_%stpr.tex' % tpr)
    common.utils.makedir(os.path.dirname(filepath))
    save_latex(filepath, latex)

    print(latex)


def distal_main_latex_table(config, config_training_names, config_attack_names, individual_adversarial_evaluations, tpr):
    """
    Distal main LaTeX table.
    """

    dataset = config.__name__.split('.')[-1].lower()

    latex = """%% %s
\\begin{tabular}{|c|}
\\hline
\\\\
\\hline
\\begin{tabular}{@{}c@{}}distal\\\\\\vphantom{t}\end{tabular}\\\\
\\hline
unseen\\\\
\\hline
FPR\\\\
\\hline
\\hline
    """ % str(datetime.datetime.now())

    for i in range(len(individual_adversarial_evaluations)):
        assert len(individual_adversarial_evaluations[i]) == 1

        if individual_adversarial_evaluations[i][0] is not None:
            latex += '%.1f %% %s fpr@%s\n' % (
                round(getattr(individual_adversarial_evaluations[i][0], 'fpr_at_%stpr' % tpr)(), 3) * 100,
                config_training_names[i],
                tpr
            )
        else:
            latex += '-- '
        latex += '\\\\\\hline\n'
    latex += '\\end{tabular}\n'

    filepath = os.path.join(dataset, 'distal_main_%stpr.tex' % tpr)
    common.utils.makedir(os.path.dirname(filepath))
    save_latex(filepath, latex)

    print(latex)


def _distal_ood_supp_latex_table(config, config_training_names, config_attack_names, individual_adversarial_evaluations, tpr):
    """
    Distal or OOD supplementary LaTeX table.
    """

    dataset = config.__name__.split('.')[-1].lower()

    latex = '%% %s\n' % str(datetime.datetime.now())
    latex += '\\begin{tabularx}{1\\textwidth}{| X |c|c|c||}\n'
    latex += '\\hline\n'
    latex += '\\multicolumn{4}{|c|}{\\textbf{%s:} Supplementary Results for Distal Adversarial Examples}\\\\\n' % dataset.upper()
    latex += '\\hline\n'
    latex += '& \\multicolumn{3}{c||}{\\begin{tabular}{c}\\textbf{Detection Setting}\\\\$\\tau$@$%s\%%$TPR\\end{tabular}}\\\\\n' % tpr
    latex += '\\hline\n'
    latex += 'Training & \\begin{tabular}{c}ROC\\\\AUC\\end{tabular} & \\begin{tabular}{c}FPR\\\\ in \\%\\end{tabular} & $\\tau$\\\\\n'
    latex += '\\hline\n'
    latex += '\\hline\n'
    for j in range(len(individual_adversarial_evaluations[0])):
        latex += '\multirow{%d}{*}{%s} ' % (len(individual_adversarial_evaluations), config_attack_names[j])
        for i in range(len(individual_adversarial_evaluations)):
            if individual_adversarial_evaluations[i][j] is not None:
                latex += '& %s & %.2f & %.1f & %.1f\\\\\n' % (
                    config_training_names[i],
                    round(individual_adversarial_evaluations[i][j].receiver_operating_characteristic_auc(), 2),
                    round(getattr(individual_adversarial_evaluations[i][j], 'fpr_at_%stpr' % tpr)(), 3) * 100,
                    round(getattr(individual_adversarial_evaluations[i][j], 'confidence_at_%stpr' % tpr)(), 2),
                )
            else:
                latex += '& %s & -- & -- & --\\\\\n' % (
                    config_training_names[i],
                )
        latex += '\\hline\n'
    latex += '\\end{tabularx}\n'
    return latex


def distal_supp_latex_table(config, config_training_names, config_attack_names, individual_adversarial_evaluations, tpr):
    """
    Distal supplementary LaTeX table.
    """

    latex = _distal_ood_supp_latex_table(config, config_training_names, config_attack_names, individual_adversarial_evaluations, tpr)

    dataset = config.__name__.split('.')[-1].lower()
    filepath = os.path.join(dataset, 'distal_supp_%stpr.tex' % tpr)
    common.utils.makedir(os.path.dirname(filepath))
    save_latex(filepath, latex)

    print(latex)