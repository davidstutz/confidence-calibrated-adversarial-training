# Confidence-Calibrated Adversarial Training

This repository contains the PyTorch code for **confidence-calibrated adversarial training (CCAT)**
corresponding to the following paper:

D. Stutz, M. Hein, B. Schiele.
**Confidence-Calibrated Adversarial Training: Generalizing to Unseen Attacks**.
ICML, 2020.

Please cite as:

    @article{Stutz2020ICML,
        author    = {David Stutz and Matthias Hein and Bernt Schiele},
        title     = {Confidence-Calibrated Adversarial Training: Generalizing to Unseen Attacks},
        journal   = {Proceedings of the International Conference on Machine Learning {ICML}},
        year      = {2020}
    }

Also check the [project page](https://davidstutz.de/projects/confidence-calibrated-adversarial-training/).

The repository allows to reproduce the experiments reported in
the paper or use training procedures and attacks as standalone components.
Features include adversarial [1] and confidence-calibrated adversarial training
on MNIST, SVHN and Cifar10, as well as `L_p` PGD [1] attacks with backtracking
and various objectives.

![Confidence-Calibrated Adversarial Training.](screenshot.jpg?raw=true "Confidence-Calibrated Adversarial Training.")

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Downloads](#downloads)
* [Examples](#examples)
* [Standalone Attacks, Training and Evaluation](#standalone-attacks-training-and-evaluation)
    * [Attacks](#attacks)
    * [Training](#training)
    * [Evaluation](#evaluation)
* [Reproduce Experiments](#reproduce-experiments)
    * [Reproduce Training](#reproduce-training)
    * [Reproduce Attacks](#reproduce-attacks)
    * [Reproduce Evaluation](#reproduce-evaluation)
    * [Reproduce Baselines](#reproduce-baselines)
* [References](#references)
* [License](#license)

## Features

This repository includes (with links to the respective parts):

* [Training procedures](common/train/README.md) for:
    * [Adversarial Training [1]](common/train/adversarial_training.py)
    * [Confidence-Calibrated Adversarial Training](common/train/confidence_calibrated_adversarial_training.py)
* [Various white- and black-box adversarial attacks](attacks/README.md):
    * [PGD [1] with backtracking](attacks/batch_gradient_descent.py)
    * (Reference implementation of PGD without backtracking)
    * [Corner Search [2]](attacks/batch_corner_search.py)
    * [Query Limited [3] with backtracking](attacks/batch_query_limited.py)
    * [ZOO [4] with backtracking](attacks/batch_zoo.py)
    * [Adversarial Frames [5]](attacks/batch_frames.py)
    * [Geometry [6]](attacks/batch_geometry.py)
    * [Square [7]](attacks/batch_cube2.py)
    * [Random sampling](attacks/batch_random.py)
* Confidence-thresholded evaluation protocol for:
    * [adverarial examples](common/eval/adversarial_evaluation.py)
    * [distal adversarial examples](common/eval/distal_evaluation.py)
    * [corrupted examples](common/eval/corrupted_evaluation.py)
* [Models](models/README.md):
    * (wide, pre-activation) ResNet
    * LeNet
    * Multilayer-perceptrons

More features:

* All attacks follow a common interface.
* All attacks allow different objectives, initialization protocols and
all `L_p` norms for `p in {infty, 2, 1, 0}`.
* All attacks can be run on individual examples or batches of examples.
* Adversarial training supports any of the included attacks and using
variable (for example 100% or 50%) fraction of adversarial examples
per batch.
* Confidence-calibrated adversarial training supports any of the included
attacks, different losses and transition functions.
* Training supports data augmentation through
[imgaug](https://imgaug.readthedocs.io/en/latest/).
* Training supports custom data loaders.
* Evaluation includes per-example worst-case analysis and multiple
restarts per attack.
* **Utilities, attacks and training are tested!**

## Installation

This repository requires, among others, the following packages:

* Python >=3.5
* [PyTorch](https://pytorch.org/) >= 1.1 and [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)
* [Tensorflow](https://www.tensorflow.org/) for [Tensorboard](https://www.tensorflow.org/tensorboard)
* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [h5py](https://www.h5py.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [scipy](https://www.scipy.org/)
* [imageio](https://imageio.github.io/)
* [imgaug](https://scikit-image.org/docs/dev/api/skimage.html)
* [iPython](https://ipython.org/) and [Jupyter](https://jupyter.org/install) (for evaluation)
* [wget](https://pypi.org/project/wget/) (for examples)

Running

    python3 setup.py

can be used to check whether all requirements are met.
The script also checks paths to data and experiments required for
reproducing the experiments.

**Without Tensorboard:** The code can be used without Tensorflow and Tensorboard by not relying
on Tensorboard to monitor training. For reproducing the experiments without
Tensorboard, remove the Tensorboard writer in `experiments/config/common.py`
(in `__get_training_writer`).

## Downloads

Datasets are provided in HDF5 format, however, can also be converted manually
as described in [Reproduce Experiments](#reproduce-experiments). Models are provided
individually (for each dataset) or together with the correct directory structure
to reproduce experiments. All models can be loaded using `common.state` as outlined
below.

**Datasets:** Datasets have been converted to HDF5 and scaled to `[0,1]`. Each dataset is plit
into four files: `train_images.h5`, `train_labels.h5`, `test_images.h5`, `test_labels.h5`.
These can be downloaded below:

<table>
    <tr>
        <th>Dataset</th>
        <th>Download</th>
    </tr>
    <tr>
        <td>MNIST</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/mnist.zip">mnist.zip</a></td>
    </tr>
    <tr>
        <td>SVHN</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/svhn.zip">svhn.zip</a></td>
    </tr>
    <tr>
        <td>Cifar10</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/cifar10.zip">cifar10.zip</a></td>
    </tr>
</table>

The datasets can easily be downloaded using the following example. Make sure that `BASE_DATA` in
`common.paths` is set to an existing directory to save the datasets in:

    # examples/readme/download_dataset.py
    
    import wget
    import zipfile
    import argparse
    import common.paths
    import common.datasets


    parser = argparse.ArgumentParser(description='Download and load a dataset')
    parser.add_argument('dataset', type=str, help='cifar10 | cifar10_c | svhn | mnist | mnist_c')
    args = parser.parse_args()
    assert args.dataset in ['cifar10', 'cifar10_c', 'svhn', 'mnist', 'mnist_c']

    # URL to fetch individual model.
    filename = '%s.zip' % args.dataset
    if not os.path.exists(filename):
        url = 'https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/%s.zip' % args.dataset
        filename = wget.download(url)

    # Directory to extract the hdf5 files to.
    dataset_dir = ''
    if args.dataset == 'mnist':
        dataset_dir = os.path.dirname(common.paths.mnist_train_images_file())
    elif args.dataset == 'mnist_c':
        dataset_dir = common.paths.raw_mnistc_dir()
    elif args.dataset == 'svhn':
        dataset_dir = os.path.dirname(common.paths.svhn_train_images_file())
    elif args.dataset == 'cifar10':
        dataset_dir = os.path.dirname(common.paths.cifar10_train_images_file())
    elif args.dataset == 'cifar10_c':
        dataset_dir = common.paths.raw_cifar10c_dir()
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    # Load dataset.
    trainset = None
    if args.dataset == 'mnist':
        trainset = common.datasets.MNISTTrainSet()
        testset = common.datasets.MNISTTestSet()
    elif args.dataset == 'mnist_c':
        testset = common.datasets.MNISTCTestSet()
    elif args.dataset == 'svhn':
        trainset = common.datasets.SVHNTrainSet()
        testset = common.datasets.SVHNTestSet()
    elif args.dataset == 'cifar10':
        trainset = common.datasets.Cifar10TrainSet()
        testset = common.datasets.Cifar10TestSet()
    elif args.dataset == 'cifar10_c':
        testset = common.datasets.Cifar10CTestSet()

    if trainset is not None:
        print('Training examples: %d' % len(trainset))
    print('Test examples: %d' % len(testset))

**Models:** The models are provided in `.pth.tar` format. Individual models are provided, as well as
all model bundled in the correct directory structure for reproducing experiments from the paper.

<table>
    <tr>
        <th>Dataset</th>
        <th>Model</th>
        <th>Download</th>
    </tr>
    <tr>
        <td colspan="2"><b>All for Reproduction</b></td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/experiments.zip">experiments.zip</a></td>
    </tr>
    <tr>
        <td>MNIST</td>
        <td>Normal</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/mnist_normal.zip">mnist_normal.zip</a></td>
    </tr>
    <tr>
        <td>MNIST</td>
        <td>AT [1]</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/mnist_at.zip">mnist_at.zip</a></td>
    </tr>
    <tr>
        <td>MNIST</td>
        <td>CCAT</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/mnist_ccat.zip">mnist_ccat.zip</a></td>
    </tr>
    <tr>
        <td>MNIST</td>
        <td>MSD [8]</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/mnist_msd.zip">mnist_msd.zip</a></td>
    </tr>
    <tr>
        <td>SVHN</td>
        <td>Normal</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/svhn_normal.zip">svhn_normal.zip</a></td>
    </tr>
    <tr>
        <td>SVHN</td>
        <td>AT [1]</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/svhn_at.zip">svhn_at.zip</a></td>
    </tr>
    <tr>
        <td>SVHN</td>
        <td>CCAT</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/svhn_ccat.zip">svhn_ccat.zip</a></td>
    </tr>
    <tr>
        <td>Cifar10</td>
        <td>Normal</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/cifar10_normal.zip">cifar10_normal.zip</a></td>
    </tr>
    <tr>
        <td>Cifar10</td>
        <td>AT [1]</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/cifar10_at.zip">cifar10_at.zip</a></td>
    </tr>
    <tr>
        <td>Cifar10</td>
        <td>CCAT</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/cifar10_ccat.zip">cifar10_ccat.zip</a></td>
    </tr>
    <tr>
        <td>Cifar10</td>
        <td>MSD [8]</td>
        <td><a href="https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/cifar10_msd.zip">cifar10_msd.zip</a></td>
    </tr>
</table>

The models can easily be downloaded using the following example.
Models are saved and loaded using `common.state`. While the models
can also be loaded using `torch.load`, `common.state` does not require to know the
used architecture in advance, as shown in the below example:

    # examples/readme/download_model.py

    import wget
    import zipfile
    import argparse
    import common.state

    parser = argparse.ArgumentParser(description='Download and load a model.')
    parser.add_argument('dataset', type=str, help='cifar10 | svhn | mnist')
    parser.add_argument('model', type=str, help='normal | at | ccat | msd')
    args = parser.parse_args()
    assert args.dataset in ['cifar10', 'svhn', 'mnist']
    assert args.model in ['normal', 'at', 'ccat', 'msd']
    if args.model == 'msd':
        assert args.dataset in ['cifar10', 'mnist']

    # URL to fetch individual model.
    url = 'https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/%s_%s.zip' % (args.dataset, args.model)
    filename = wget.download(url)

    # Directory to extract the model to.
    model_dir = './%s_%s/' % (args.dataset, args.model)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

    # Path to model file.
    model_file = '%s/classifier.pth.tar' % model_dir
    assert os.path.exists(model_file)

    # Loading using common.state without knowing the architecture.
    state = common.state.State.load(model_file)
    model = state.model
    print(model)

## Examples

**Examples:** in [`examples/`](examples/README.md):

* `normal_training_robustness.py`: robustness evaluation for normal training.
* `adversarial_training_robustness.py`: robustness evaluation for _adversarial training_.
* `confidence_calibrated_adversarial_training_robustness.py`: robustness evaluation for
_confidence-calibrated adversarial training_.

Examples from this README in `examples/readme/`:

* `download_dataset.py`: download and load datasets converted to HDF5;
* `download_model.py`: download and load models;
* `run_attacks.py`: running attacks;
* `run_lp_attacks.py`: running various `L_p` attacks;
* `run_distal_attacks.py`: running distal attacks;
* `train_normal.py`: normal training on MNIST;
* `train_adversarial.py`: adversarial training on MNIST;
* `train_confidence_calibrated.py`: confidence-calibrated adversarial training on MNIST;
* `evaluate_adversarial.py`: evaluate adversarial robustness, including worst-case and confidence-thresholded evaluation;
* `evaluate_distal.py`: evaluate against distal adversarial examples;
* `evaluate_corruptedcorrupted_evaluation.py`: evaluate on corrupted examples;

Tests can be found in [`tests/`](tests/README.md) and also contain many usage examples.

## Standalone Attacks, Training and Evaluation

The repository is organized in a very modular way, including the following components:

* `attacks`: standalone attacks, depending only on some utilities in `common`:
    * Applicable to any PyTorch model;
    * Allowing various `L_p` norms and objectives;
    * Applicable to batches of images;
* `common`: common utilities, used for attacks, training and evaluation;
* `common.train`: trainers for normal, adversarial and
confidence-calibrated adversarial training:
    * Allowing to train any PyTorch model;
    * Applicable to any dataset that can be wrapped in `torch.utils.data.DataLoader`;
    * Flexible data augmentation using [imgaug](https://scikit-image.org/docs/dev/api/skimage.html).
* `common.eval`: adversarial evaluation including:
    * Unthresholded (robust) test error, ROC AUC;
    * Confidence-thresholded (robust) test error, false positive rate;

### Attacks

All attacks implement the following abstract class:

    class Attack:
        def __init__(self):
            """
            Constructor, setting initial attributes, potentially excepting some hyper-parameters.
            """
            pass
        def run(self, model, images, objective, writer=common.summary.SummaryWriter(), prefix=''):
            """
            Run the attack on the given model and batch of images.
            The objective has to follow attacks.objectives.Objective and
            can be the cross-entropy loss; all attacks implement
            a minimization problem.
            """
            pass

The repository includes several objectives; each objective implements an error
that is to be minimized by implementing the following abstract class:

    class Objective:
        def __call__(self, logits, perturbations=None):
            """
            Computes the objective, given the logits (pre-softmax predictions
            of model) and the corresponding perturbations (if necessary).
            """
            pass

Before running an attack on a batch of images, the true labels (and target labels for targeted attacks)
need to be set using the objective:

    objective.set(true_classes, target_classes) # Both might be None

Given model, attack, objective and a dataloader, `common.test` can be used to easily run an attack
and obtain adversarial perturbations, the corresponding predicted probabilities and errors. This is
shown in the following example (see `examples/readmer/attacks.py`):

    # examples/readme/run_attacks.py
    
    import torch
    import attacks
    import common.state
    import common.test
    import common.datasets
    import common.eval


    # Load a pre-trained normal or adversarial training model
    model_file = 'mnist_ccat/classifier.pth.tar'
    # common.state.State will automatically determine the corresponding architecture
    state = common.state.State.load(model_file)
    model = state.model

    cuda = True
    if cuda:
        model = model.cuda()

    # Test set and data loader for 1000 images of MNIST
    batch_size = 100
    testset = common.datasets.MNISTTestSet(indices=range(100))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Set up a basic PGD attack with 40 iterations, maximizing cross-entropy loss
    epsilon = 0.3
    attack = attacks.BatchGradientDescent()
    attack.max_iterations = 40
    attack.base_lr = 0.05
    attack.momentum = 0.9  # use momentum
    attack.c = 0
    attack.lr_factor = 1.5
    attack.normalized = True  # use signed gradient
    attack.backtrack = True  # use bactracking
    # Adversarial examples are initialized randomly within L_inf epsilon ball
    attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
    # Adversarial examples are projected onto [0, 1] box and L_inf epsilon ball
    attack.projection = attacks.projections.SequentialProjections([
        attacks.projections.LInfProjection(epsilon),
        attacks.projections.BoxProjection()
    ])
    attack.norm = attacks.norms.LInfNorm()
    # Maximize cross-entropy loss (i.e., minimize minus cross-entropy loss)
    objective = attacks.objectives.UntargetedF0Objective()

    model.eval()

    # Evaluate model on clean test set
    clean_probabilities = common.test.test(model, testloader, cuda=cuda)
    # Attack test set, allowing one random restart only
    adversarial_perturbations, adversarial_probabilities, adversarial_errors = common.test.attack(model, testloader, attack,
                                                                                                  objective, attempts=5, cuda=cuda)
    # Evaluation without confidence thresholding.
    eval = common.eval.AdversarialEvaluation(clean_probabilities, adversarial_probabilities, testset.labels, validation=0)
    print('robust test error in %%: %g' % eval.robust_test_error())
 
**`L_p` Variants:** The provided variant of projected gradient descent (PGD) with momentum and backtracking is an effective
attack to be used in various `L_p` norms using different initializations. As shown in the example above,
the attack has the following parameters:

    # examples/readme/run_lp_attacks.py

    import torch
    import attacks
    import common.state
    import common.test
    import common.datasets
    import common.eval


    model_file = 'mnist_ccat/classifier.pth.tar'
    state = common.state.State.load(model_file)
    model = state.model

    cuda = True
    if cuda:
        model = model.cuda()

    batch_size = 100
    testset = common.datasets.MNISTTestSet()
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    adversarialset = common.datasets.MNISTTestSet(indices=range(100))
    adversarialloader = torch.utils.data.DataLoader(adversarialset, batch_size=batch_size, shuffle=False, num_workers=0)

    linf_epsilon = 0.3
    linf_attack = attacks.BatchGradientDescent()
    linf_attack.max_iterations = 40
    linf_attack.base_lr = 0.05
    linf_attack.momentum = 0.9
    linf_attack.c = 0
    linf_attack.lr_factor = 1.5
    linf_attack.normalized = True
    linf_attack.backtrack = True
    linf_attack.initialization = attacks.initializations.LInfUniformNormInitialization(linf_epsilon)
    linf_attack.projection = attacks.projections.SequentialProjections([
        attacks.projections.LInfProjection(linf_epsilon),
        attacks.projections.BoxProjection()
    ])
    linf_attack.norm = attacks.norms.LInfNorm()

    l2_epsilon = 3
    l2_attack = attacks.BatchGradientDescent()
    l2_attack.max_iterations = 40
    l2_attack.base_lr = 0.05
    l2_attack.momentum = 0.9
    l2_attack.c = 0
    l2_attack.lr_factor = 1.5
    l2_attack.normalized = True
    l2_attack.backtrack = True
    l2_attack.initialization = attacks.initializations.L2UniformNormInitialization(l2_epsilon)
    l2_attack.projection = attacks.projections.SequentialProjections([
        attacks.projections.L2Projection(l2_epsilon),
        attacks.projections.BoxProjection()
    ])
    l2_attack.norm = attacks.norms.L2Norm()

    l1_epsilon = 18
    l1_attack = attacks.BatchGradientDescent()
    l1_attack.max_iterations = 40
    l1_attack.base_lr = 0.5
    l1_attack.momentum = 0.9
    l1_attack.c = 0
    l1_attack.lr_factor = 1.5
    l1_attack.normalized = True
    l1_attack.backtrack = True
    l1_attack.initialization = attacks.initializations.L1UniformNormInitialization(l1_epsilon)
    l1_attack.projection = attacks.projections.SequentialProjections([
        attacks.projections.L1Projection(l1_epsilon),
        attacks.projections.BoxProjection()
    ])
    l1_attack.norm = attacks.norms.L1Norm()

    l0_epsilon = 15
    l0_attack = attacks.BatchGradientDescent()
    l0_attack.max_iterations = 40
    l0_attack.base_lr = 250
    l0_attack.momentum = 0.9
    l0_attack.c = 0
    l0_attack.lr_factor = 1.5
    l0_attack.normalized = True
    l0_attack.backtrack = True
    l0_attack.initialization = attacks.initializations.L1UniformNormInitialization(l0_epsilon)
    l0_attack.projection = attacks.projections.SequentialProjections([
        attacks.projections.L0Projection(l0_epsilon),
        attacks.projections.BoxProjection()
    ])
    l0_attack.norm = attacks.norms.L0Norm()

    objective = attacks.objectives.UntargetedF0Objective()
    labels = ['linf', 'l2', 'l1', 'l0']
    epsilons = [linf_epsilon, l2_epsilon, l1_epsilon, l0_epsilon]
    attacks = [linf_attack, l2_attack, l1_attack, l0_attack]

    model.eval()
    clean_probabilities = common.test.test(model, testloader, cuda=cuda)

    for a in range(len(attacks)):
        _, adversarial_probabilities, _ = common.test.attack(model, adversarialloader, linf_attack, objective, attempts=1, cuda=cuda)
        eval = common.eval.AdversarialEvaluation(clean_probabilities, adversarial_probabilities, testset.labels, validation=0.1)
        print('[%s, epsilon=%g] robust test error in %%: %g' % (
            labels[a],
            epsilons[a],
            (100*eval.robust_test_error())
        ))
        print('[%s, epsilon=%g] robust test error @99%%TPR in %%: %g' % (
            labels[a],
            epsilons[a],
            (100 * eval.robust_test_error_at_99tpr())
        ))

The above example shows a standard `L_inf` PGD attack for `epsilon = 0.3`, as on MNIST. Other
`L_p` attacks can easily be obtained by adapting the norm, initialization and projection:
    
    # examples/readme/lp_attacks.py
    
    # L_inf attack:
    linf_epsilon = 0.3
    linf_attack = attacks.BatchGradientDescent()
    linf_attack.max_iterations = 40
    linf_attack.base_lr = 0.05
    linf_attack.momentum = 0.9
    linf_attack.c = 0
    linf_attack.lr_factor = 1.5
    linf_attack.normalized = True
    linf_attack.backtrack = True
    linf_attack.initialization = attacks.initializations.LInfUniformNormInitialization(linf_epsilon)
    linf_attack.projection = attacks.projections.SequentialProjections([
        attacks.projections.LInfProjection(linf_epsilon),
        attacks.projections.BoxProjection()
    ])
    linf_attack.norm = attacks.norms.LInfNorm()

    # L_2 attack:
    l2_epsilon = 3
    l2_attack = attacks.BatchGradientDescent()
    l2_attack.max_iterations = 40
    l2_attack.base_lr = 0.05
    l2_attack.momentum = 0.9
    l2_attack.c = 0
    l2_attack.lr_factor = 1.5
    l2_attack.normalized = True
    l2_attack.backtrack = True
    # Note L_2 initialization, projection, norm!
    l2_attack.initialization = attacks.initializations.L2UniformNormInitialization(l2_epsilon)
    l2_attack.projection = attacks.projections.SequentialProjections([
        attacks.projections.L2Projection(l2_epsilon),
        attacks.projections.BoxProjection()
    ])
    l2_attack.norm = attacks.norms.L2Norm()

    # L_1 attack:
    l1_epsilon = 18
    l1_attack = attacks.BatchGradientDescent()
    l1_attack.max_iterations = 40
    l1_attack.base_lr = 0.5
    l1_attack.momentum = 0.9
    l1_attack.c = 0
    l1_attack.lr_factor = 1.5
    l1_attack.normalized = True
    l1_attack.backtrack = True
    # Note L_1 initialization, projection, norm!
    l1_attack.initialization = attacks.initializations.L1UniformNormInitialization(l1_epsilon)
    l1_attack.projection = attacks.projections.SequentialProjections([
        attacks.projections.L1Projection(l1_epsilon),
        attacks.projections.BoxProjection()
    ])
    l1_attack.norm = attacks.norms.L1Norm()

    # L_0 attack:
    l0_epsilon = 15
    l0_attack = attacks.BatchGradientDescent()
    l0_attack.max_iterations = 40
    l0_attack.base_lr = 250
    l0_attack.momentum = 0.9
    l0_attack.c = 0
    l0_attack.lr_factor = 1.5
    l0_attack.normalized = True
    l0_attack.backtrack = True
    # Note L_0 initialization, projection, norm!
    l0_attack.initialization = attacks.initializations.L0UniformNormInitialization(l0_epsilon)
    l0_attack.projection = attacks.projections.SequentialProjections([
        attacks.projections.L0Projection(l1_epsilon),
        attacks.projections.BoxProjection()
    ])
    l0_attack.norm = attacks.norms.L0Norm()

    objective = attacks.objectives.UntargetedF0Objective()
    labels = ['linf', 'l2', 'l1', 'l0']
    epsilons = [linf_epsilon, l2_epsilon, l1_epsilon, l0_epsilon]
    attacks = [linf_attack, l2_attack, l1_attack, l0_attack]
    clean_probabilities = common.test.test(model, testloader, cuda=cuda)

    for a in range(len(attacks)):
        _, adversarial_probabilities, _ = common.test.attack(model, testloader, linf_attack, objective, attempts=1, cuda=cuda)
        eval = common.eval.AdversarialEvaluation(adversarial_probabilities, clean_probabilities, testset.labels, validation=0)
        print('[%s, epsilon=%g] robust test error in %%: %g' % (
            labels[a],
            epsilons[a],
            eval.robust_test_error()
        ))

Similarly, many other attacks can be adapted:

* `attacks.BatchQueryLimited`
* `attack.BatchZOO`
* `attacks.BatchSimple`
* `attacks.BatchGeometry`

**Distal Adversarial Examples:** Distal adversarial examples can be computed as regular adversarial
examples when starting from random examples instead of (clean) test examples. Additionally,
as objective, _any_ logit is maximized (to obtain high-confidence distal adversarial examples):

    # examples/readme/run_distal_attacks.py
    
    import torch
    import attacks
    import common.state
    import common.test
    import common.datasets
    import common.eval


    # Load a pre-trained normal or adversarial training model
    model_file = 'mnist_ccat/classifier.pth.tar'
    # common.state.State will automatically determine the corresponding architecture
    state = common.state.State.load(model_file)
    model = state.model

    cuda = True
    if cuda:
        model = model.cuda()

    batch_size = 100
    testset = common.datasets.MNISTTestSet()
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    # Main difference to regular adversarial examples: start with random images!
    randomset = common.datasets.RandomTestSet(100, [28, 28, 1])
    randomloader = torch.utils.data.DataLoader(randomset, batch_size=batch_size, shuffle=False)

    epsilon = 0.3
    attack = attacks.BatchGradientDescent()
    attack.max_iterations = 40
    attack.base_lr = 0.05
    attack.momentum = 0.9
    attack.c = 0
    attack.lr_factor = 1.5
    attack.normalized = True
    attack.backtrack = True
    attack.initialization = attacks.initializations.RandomInitializations([
        attacks.initializations.LInfUniformNormInitialization(epsilon)
    ])
    attack.projection = attacks.projections.SequentialProjections([
        attacks.projections.LInfProjection(epsilon),
        attacks.projections.BoxProjection()
    ])
    attack.norm = attacks.norms.LInfNorm()
    # Maximize any log-softmax (logit) to obtain high confidence.
    objective = attacks.objectives.UntargetedF0Objective(loss=common.torch.max_log_loss)

    model.eval()

    clean_probabilities = common.test.test(model, testloader, cuda=cuda)
    _, distal_probabilities, _ = common.test.attack(model, randomloader, attack,
                                                         objective, attempts=5, cuda=cuda)

    eval = common.eval.DistalEvaluation(clean_probabilities, distal_probabilities,
                                        testset.labels, validation=0.1)
    print('confidence threshold @99%%TPR: %g' % eval.confidence_at_99tpr())
    print('false positive rate @99%%TPR in %%: %g' % eval.fpr_at_99tpr())

### Training

Training procedures for normal training, adversarial training and confidence-calibrated adversarial training
are provided. Each procedure implements the following abstract class, allowing easy usage:

    class TrainingInterface:
        def train(self, epoch):
            """
            Perform one training epoch; here, epoch is the current epoch.
            """
            raise NotImplementedError()
        def test(self, epoch):
            """
            Perform one test epoch.
            """
            raise NotImplementedError()
        def step(self, epoch):    
            self.train(epoch)
            self.test(epoch)

**Normal Training:** For example, training a normal model, can be done using the following snippet.
Note that training only requires a model providing a forward pass (`.forward`)
and a train and test set in the form of `torch.utils.data.DataLoader`. This can be
one of the datasets provided by PyTorch, or as included in `common.datasets`:

    # examples/readme/train_normal.py

    import math
    import torchvision
    import torch.utils.data
    import common.train


    batch_size = 100
    # Training and test set provided by torchvision.
    # Alternatively, use common.datasets here together with torch.utils.data.DataLoader.
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('../data', train=True, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Lambda(lambda x: x.view(28, 28, 1))
                           ])),
            batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('../data', train=False, transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Lambda(lambda x: x.view(28, 28, 1))
                           ])),
            batch_size=batch_size, shuffle=False)


    class Flatten(torch.nn.Module):
        def forward(self, x):
            return x.view(x.shape[0], -1)


    # Setup a model, the optimizer, learning rate scheduler.
    # No more required for common.train.NormalTraining.
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 5, padding=2), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(32, 64, 5, padding=2), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
        Flatten(),
        torch.nn.Linear(7*7*64, 1024), torch.nn.ReLU(),
        torch.nn.Linear(1024, 10)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    batches_per_epoch = len(train_loader)
    gamma = 0.97
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: gamma ** math.floor(epoch/batches_per_epoch)])
    trainer = common.train.NormalTraining(model, train_loader, test_loader, optimizer, scheduler)

    # Train for 10 epochs, each step contains an epoch of training and testing:
    epochs = 10
    for e in range(epochs):
        trainer.step(e)
        # The trainer does not create snapshots automatically!

    # Alternatively, use common.state here.
    torch.save(model.state_dict(), 'classifier.pth.tar')

**Adversarial Training:** For adversarial training [1], only an attack and an attack
objective is needed in addition:

    # examples/readme/train_adversarial.py
    
    # ...
    # example for MNIST with epsilon = 0.3
    epsilon = 0.3
    attack = attacks.BatchGradientDescent()
    attack.max_iterations = 40
    attack.base_lr = 0.05
    attack.momentum = 0.9
    attack.c = 0
    attack.lr_factor = 1.5
    attack.normalized = True
    attack.backtrack = True
    attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.LInfNorm()
    objective = attacks.objectives.UntargetedF0Objective()

    trainer = common.train.AdversarialTraining(model, train_loader, test_loader, optimizer, scheduler, attack, objective, fraction=0.5)
    # ...

**Confidence-Calibrated Adversarial Training:** For confidence-calibrated adversarial training,
only the attack objective needs to be changed. Additionally, a loss (between distributions) and the transition
needs to be added:

    # examples/readme/train_confidence_calibrated.py
    
    # example for MNIST with epsilon = 0.3
    epsilon = 0.3
    attack = attacks.BatchGradientDescent()
    attack.max_iterations = 40
    attack.base_lr = 0.005
    attack.momentum = 0.9
    attack.c = 0
    attack.lr_factor = 1.5
    attack.normalized = True
    attack.backtrack = True
    attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.LInfNorm()
    objective = attacks.objectives.UntargetedF7PObjective()

    loss = common.torch.cross_entropy_divergence
    transition = common.utils.partial(common.torch.power_transition, norm=attacks.norms.LInfNorm(), gamma=12, epsilon=0.3)
    trainer = common.train.ConfidenceCalibratedAdversarialTraining(model, train_loader, test_loader, optimizer, scheduler, attack, objective, loss, transition, fraction=0.5)

### Evaluation

Evaluation is split into the following components:

* `common.eval.CleanEvaluation`: evaluation on clean examples, for example, test error and
confidence-thresholded test error;
* `common.eval.AdversasrialEvaluation`: evaluation on adversarial examples, for example,
robust test error and confidence-thresholded robust test error; also includes ROC AUC and
false positive rate;
* `common.eval.CorruptedEvaluation`: evaluation on corrupted examples such as MNIST-C and
Cifar10-C, for example, test error and confidence-thresholded test error;
* `common.eval.DistalEvaluation`: evaluation on distal adversarial examples, for example,
ROC AUC and false positive rate;

In each cases, evaluation is based on predicted probabilities (on clean, adversarial, or
corrupted examples). The predicted probabilities on clean examples can be obtained using 
`common.test.test`. For adversarial examples, they are returned by `common.test.attack`. 
Evaluation supports multiple attempts of the attack, as also supported by `common.test.attack`:

    # examples/readme/evaluate_adversarial.py
    
    import torch
    import attacks
    import common.state
    import common.test
    import common.datasets
    import common.eval


    # Load a pre-trained normal or adversarial training model
    model_file = 'mnist_ccat/classifier.pth.tar'
    # common.state.State will automatically determine the corresponding architecture
    state = common.state.State.load(model_file)
    model = state.model

    cuda = True
    if cuda:
        model = model.cuda()

    batch_size = 100
    testset = common.datasets.MNISTTestSet()
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    adversarialset = common.datasets.MNISTTestSet(indices=range(100))
    adversarialloader = torch.utils.data.DataLoader(adversarialset, batch_size=batch_size, shuffle=False)

    epsilon = 0.3
    attack = attacks.BatchGradientDescent()
    attack.max_iterations = 40
    attack.base_lr = 0.05
    attack.momentum = 0.9  # use momentum
    attack.c = 0
    attack.lr_factor = 1.5
    attack.normalized = True
    attack.backtrack = True
    attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
    attack.projection = attacks.projections.SequentialProjections([
        attacks.projections.LInfProjection(epsilon),
        attacks.projections.BoxProjection()
    ])
    attack.norm = attacks.norms.LInfNorm()
    objective = attacks.objectives.UntargetedF0Objective()

    model.eval()

    # Obtain predicted probabilities on clean examples.
    # Note that the full test set is used.
    clean_probabilities = common.test.test(model, testloader, cuda=cuda)
    # Run attack, will also provide the corresponding predicted probabilities.
    # Note that only 1000 examples are attacked.
    _, adversarial_probabilities, adversarial_errors = common.test.attack(model, adversarialloader, attack,
                                                         objective, attempts=5, cuda=cuda)
    print(clean_probabilities.shape) # 10000 x 10
    print(adversarial_probabilities.shape) # 5 x 100 x 10

    # Use validation=0.1 such that 10% of the clean probabilities are
    # used to determine the confidence threshold.
    eval = common.eval.AdversarialEvaluation(clean_probabilities, adversarial_probabilities,
                                             testset.labels, validation=0.1, errors=adversarial_errors)
    print('test error in %%: %g' % eval.test_error())
    print('robust test error in %%: %g' % eval.robust_test_error())

    print('confidence threshold @99%%TPR: %g' % eval.confidence_at_99tpr())
    print('test error @99%%TPR in %%: %g' % eval.test_error_at_99tpr())
    print('false positive rate @99%%TPR in %%: %g' % eval.fpr_at_99tpr())
    print('robust test error @99%%TPR in %%: %g' % eval.robust_test_error_at_99tpr())

The above example illustrates how to compute confidence-thresholded metrics such as the
robust test error. `common.eval.AdversarialEvaluation` works as follows:

    def __init__(self, clean_probabilities, adversarial_probabilities, labels, validation=0.1, errors=None, include_misclassifications=False, detector=common.numpy.max_detector, clean_scores=None, adversarial_scores=None):
        """
        Adversarial evaluation.
        
        :param clean_probabilities: probabilities on clean examples
        :type clean_probabilities: numpy.ndarray
        :param adversarial_probabilities: probabilities on adversarial examples
        :type adversarial_probabilities: numpy.ndarray
        :param labels: labels
        :type labels: numpy.ndarray
        :param validation: fraction of validation examples
        :type validation: float
        :param errors: errors to determine worst case
        :type errors: None or numpy.ndarray
        :param include_misclassifications: include mis classifications in confidence threshold computation
        :type include_misclassifications: bool
        :param detector: detector to apply on probabilities (default is taking the maximum confidence)
        :type detector: callable
        :param clean_scores: scores corresponding to clean_probabilities
        :type clean_scores: numpy.ndarray
        :param adversarial_scores: scores corresponding to clean_probabilities
        :type adversarial_scores: numpy.ndarray
        """
        
Here, the following dimensions are expected:

* `clean_probabilities`: predicted probabilities on clean examples, a `N_1 x K` array where `N_1` is the number of test examples and `K` the number of classes
* `adversarial_probabilities`: predicted probabilities on adversarial examples, a `A x N_2 x K` where `A` is the number of attempts (of the attacks), `N_2 < N_1` the number of attacked test examples
* `labels`: labels of test examples, a `N_1` array
* `errors`: error values of attacks (lower error means stronger attack), a `A x N_2` array where each element is the error corresponding to the adversarial probabilities in `adversarial_probabilities`
* `clean_scores`: detection scores (if `detector` is None) on clean examples, a `N_1` array
* `adversarial_scores`: detection scores (if `detector` is None) on adversarial examples, a `A x N_2` array

Then, adversarial evaluation operates in two modes:

* If `errors` are **not** provided, all attempts (i.e., A for the array) are treated as individual attacks; this means that the metrics are averages over all attempts.
* If `errors` are provided, the worst-case attempt is selected for evaluation.

For confidence-thresholded evaluation ...

* `detector`, if not None, is applied on clean and adversarial probabilities to determine confidences on which a threshold is chosen and evaluation;
* `clean_scores` and `adversarial_scores`, if `detector` is None, are used as scores (i.e., confidences) for choosing a threshold and evaluation;

The threshold is chosen according to the true positive rate. Here, positives are correctly classified clean examples (unless `include_misclassifications` is True)
and negatives are successful adversarial examples corresponding to correctly classified clean examples. For determining the threshold for a specific
true positive rate, the confidences (or scores) of correctly classified clean examples are sorted and the threshold is chosen to ensure at least the chosen
true positive rate. This is done on the last `validation` percent of the provided clean probabilities/scores.

Evaluation of distal adversarial examples with `common.eval.DistalEvaluation` works similarly:

    # examples/readme/evaluate_distal.py
    
    import torch
    import attacks
    import common.state
    import common.test
    import common.datasets
    import common.eval


    # Load a pre-trained normal or adversarial training model
    model_file = 'mnist_ccat/classifier.pth.tar'
    # common.state.State will automatically determine the corresponding architecture
    state = common.state.State.load(model_file)
    model = state.model

    cuda = True
    if cuda:
        model = model.cuda()

    batch_size = 100
    testset = common.datasets.MNISTTestSet()
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    # Distal adversarial examples are computed on random "images".
    adversarialset = common.datasets.RandomTestSet(100, size=(28, 28, 1))
    adversarialloader = torch.utils.data.DataLoader(adversarialset, batch_size=batch_size, shuffle=False)

    epsilon = 0.3
    attack = attacks.BatchGradientDescent()
    attack.max_iterations = 40
    attack.base_lr = 0.05
    attack.momentum = 0.9
    attack.c = 0
    attack.lr_factor = 1.5
    attack.normalized = True
    attack.backtrack = True
    attack.initialization = attacks.initializations.RandomInitializations([
        attacks.initializations.LInfUniformNormInitialization(epsilon)
    ])
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.LInfNorm()
    objective = attacks.objectives.UntargetedF0Objective(loss=common.torch.max_log_loss)

    model.eval()

    # Obtain predicted probabilities on clean examples.
    # Note that the full test set is used.
    clean_probabilities = common.test.test(model, testloader, cuda=cuda)
    # Run attack, will also provide the corresponding predicted probabilities.
    # Note that only 1000 examples are attacked.
    _, adversarial_probabilities, adversarial_errors = common.test.attack(model, adversarialloader, attack,
                                                         objective, attempts=5, cuda=cuda)
    print(clean_probabilities.shape) # 10000 x 10
    print(adversarial_probabilities.shape) # 5 x 1000 x 10

    # Use validation=0.1 such that 10% of the clean probabilities are
    # used to determine the confidence threshold.
    eval = common.eval.DistalEvaluation(clean_probabilities, adversarial_probabilities,
                                        testset.labels, validation=0.1, errors=adversarial_errors)
    print('confidence threshold @99%%TPR: %g' % eval.confidence_at_99tpr())
    print('false positive rate @99%%TPR in %%: %g' % eval.fpr_at_99tpr())

For evaluating corrupted examples, `common.datasets` provides individual corruptions of
MNIST-C and Cifar10-C:

    # examples/readme/evaluate_corrupted.py
    
    import torch
    import common.state
    import common.test
    import common.datasets
    import common.eval


    # Load a pre-trained normal or adversarial training model
    model_file = 'mnist_ccat/classifier.pth.tar'
    # common.state.State will automatically determine the corresponding architecture
    state = common.state.State.load(model_file)
    model = state.model

    cuda = True
    if cuda:
        model = model.cuda()

    batch_size = 100
    testset = common.datasets.MNISTTestSet()
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    corruptions = [
        'brightness',
        'canny_edges',
        'dotted_line',
        'fog',
        'glass_blur',
        'impulse_noise',
        'motion_blur',
        'rotate',
        'scale',
        'shear',
        'shot_noise',
        'spatter',
        'stripe',
        'translate',
        'zigzag'
    ]

    # Get dataloaders for individual corruptions.
    corrupted_loaders = []
    for i in range(len(corruptions)):
        corrupted_loaders.append(torch.utils.data.DataLoader(common.datasets.MNISTCTestSet(corruptions=[corruptions[i]], indices=list(range(1000))),
                                                      batch_size=batch_size, shuffle=False, num_workers=0))

    model.eval()

    # Evaluate corruptions individually.
    clean_probabilities = common.test.test(model, testloader, cuda=cuda)
    for i in range(len(corruptions)):
        corrupted_probabilities = common.test.test(model, corrupted_loaders[i], cuda=cuda)
        corrupted_probabilities = corrupted_probabilities.reshape(len(corrupted_loaders[i].dataset.corruptions), -1, corrupted_probabilities.shape[1])
        eval = common.eval.CorruptedEvaluation(clean_probabilities, corrupted_probabilities, testset.labels, validation=0.1)
        print(corruptions[i])
        print('confidence threshold @99%%TPR: %g' % eval.confidence_at_99tpr())
        print('test error @99%%TPR in %%: %g' % eval.test_error_at_99tpr())

## Reproduce Experiments

Reproducing the experiments is simple as long as the paths in `common/paths.py` are set correctly
and datasets and pre-trained models have been downloaded from [Downloads](#downloads):

**Setup:** The base directories for experiments and data need to be adapted in `common/paths.py`:

    # will contain mnist/, Cifar10/, and svhn/ subdirectories:
    BASE_DATA = '/absolute/path/to/data/'
    # downloaded pre-trained models can be put here:
    BASE_EXPERIMENTS = '/absolute/path/to/experiments/'
    # contains log files if necessary (mostly for training)
    BASE_LOGS = '/absolute/path/to/logs/or/tmp/'

Ideally, these should be absolute paths. The data directory should be the parent directory of the
downloaded datasets. The models and adversarial examples will be stored in the
experiments directory. The log directory will mainly contain tensorboard logs (if used),
which can get very big for longer training.

**Datasets:** The datasets can be converted using the scripts in `data/`. Alternatively, the datasets
can be downloaded from [Downloads](#downloads). The datasets are provided in HDF5 format,
individual files contain train/test images/labels. All images are normalized in `[0,1]`. The downloaded
datasets should be saved in `BASE_DATA`, for example, such that `BASE_DATA/mnist/train_images.h5` exists.

**Models:** Models can also be downloaded from [Downloads](#downloads). For reproducing the experiments,
all models (for all datasets) are offered in a single ZIP file, using the directory structure as required
below. The files should be extracted in `BASE_EXPERIMENTS`, for example, such tath 
`BASE_EXPERIMENTS/MNIST/normal_training_check` exists.

**Overview:** The experiments are defined in `experiments/config`. For example,
`experiments/config.cifar10.py` contains the hyper-parameters for training and evaluatioin on Cifar10.
`experiments/config.common.py` contain details on the attacks used for evaluation and the trained models.
The attacks and models defined in `experiments/config/common.py` can be used through the command
line tools `experiments/train.py` and `experiments/attack.py`.

### Training

The following models can be trained:

* `confidence_calibrated_adversarial_training_ce_f7p_i40_random_momentum_backtrack_power2_10`: Confidence-calibrated adversarial training (CCAT) with the power transition and `rho = 10` as used in the paper;
* `adversarial_training_lr005_f7p_i40_half_momentum_backtrack_check`: Adversarial training (AT) using 50% clean and 50% adversarial examples;
* `normal_training_check`: Normal training as reference.

Training can be started using:

    python3 train.py <training_interface> config.<dataset> <model>
    
For example, on SVHN, to train our CCAT model:

    python3 train.py ConfidenceCalibratedAdversarialTrainingInterface config.svhn confidence_calibrated_adversarial_training_ce_f7p_i40_random_momentum_backtrack_power2_10

Training can be monitored using

    python3 train_board.py config.<dataset> <model> --port <port>

which will start a TensorBoard session on the provided port.

The training interface will be 

* `ConfidenceCalibratedAdversarialTrainingInterface` for confidence-calibrated adversarial training;
* `AdversarialTrainingInterface` for adversarial training;
* `NormalTrainingInterface` for normal training.

### Attacks

The following sets of attacks are provided; see above for a list of included attacks with references.

* `set_linf_white`: `L_infty` white-box attacks, including PGD-CE and PGD-Conf as described in the paper;
* `set_inf_black`: `L_infty` black-box attacks;
* `set_lp_white`: `L_p` white-box attacks for `p` in `{0, 1, 2}`;
* `set_lp_black`: `L_p` black-box attacks for `p` in `{0, 1, 2}`;
* ...
* `set_linf_acet`: distal adversarial examples;
* `set_frames`: adversarial frames maximizing cross entropy or confidence;

(For `L_2`, `set_l2_12e3_white` and `set_l2_12e3_black` where used for the epsilon used in the paper, similarly, `set_l1_24_white` and `set_l1_24_black` where used for `L_1`.)

These sets are defined at the end of `experiments/config.common.py`.

Attacking can be started using:

    python3 attack.py config<dataset> <attack>
    
Both individual attacks as well as sets of attacks can be used, for example:

    python3 attack.py config.svhn confidence_calibrated_adversarial_training_ce_f7p_i40_random_momentum_backtrack_power2_10 set_linf_white
    python3 attack.py config.svhn confidence_calibrated_adversarial_training_ce_f7p_i40_random_momentum_backtrack_power2_10 normalized_zero_pgd_50_f7p_0001_momentum_backtrack

where `normalized_zero_pgd_50_f7p_0001_momentum_backtrack` is our PGD-Conf attack with zero initialization.

For detailed definitions of the attacks, please refer to `experiments/config/common.py`.

### Evaluation

After training and attacking, evaluation is done in Jupyter notebooks, these are found in `experiments/eval/`.

For example, `experiments/eval/main.ipynb` will produce the main results on SVHN by default; the Jupyter notebooks contain some explanations and comments and only few changes allow to evaluate on MNIST and Cifar10 as well.

### Baselines

**MSD [8]:** MSD can be evaluated as described above; however, training is not possible.

**Mahalanobis and LID Detectors:** The Mahalanobis [9] and local intrinsic dimensionality (LID) [10]
detectors were evaluated based on the code provided in
[pokaxpoka/deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector).
The corresponding code will be provided in a separate repository, see the
[project page](https://davidstutz.de/projects/adversarial-training/).

## References

    [1] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A.
        Towards deep learning models resistant to adversarial attacks.
        ICLR, 2018.
    [2] Croce, F. and Hein, M.
        Sparse and imperceivable adversarial attacks.
        arXiv.org, abs/1909.05040, 2019.
    [3] Ilyas, A., Engstrom, L., Athalye, A., and Lin, J.
        Black-box adversarial attacks with limited queries and information.
        In ICML, 2018.
    [4] Pin-Yu Chen, Huan Zhang, Yash Sharma, Jinfeng Yi, Cho-Jui Hsieh.
        ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models.
        AISec@CCS, 2017.
    [5] Zajac, M., Zolna, K., Rostamzadeh, N., and Pinheiro, P. O.
        Adversarial framing for image and video classification.
        In AAAI Workshops, 2019.
    [6] Khoury, M. and Hadfield-Menell, D.
        On the geometry of adversarial examples.
        arXiv.org, abs/1811.00525, 2018.
    [7] Andriushchenko, M., Croce, F., Flammarion, N., and Hein, M.
        Square attack: a query-efficient black-box adversarial attack via random search.
        arXiv.org, 1912.00049, 2019.
    [8] Pratyush Maini, Eric Wong, J. Zico Kolter.
        Adversarial Robustness Against the Union of Multiple Perturbation Models.
        CoRR abs/1909.04068 (2019).

## License

This repository includes code from:

* [max-andr/square-attack](https://github.com/max-andr/square-attack)
* [fra31/sparse-imperceivable-attacks](https://github.com/fra31/sparse-imperceivable-attacks)
* [gist.github.com/daien](https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246)
* [ftramer/MultiRobustness](https://github.com/ftramer/MultiRobustness)
* [meliketoy/wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch)
* [pokaxpoka/deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector)

Copyright (c) 2020 David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.
