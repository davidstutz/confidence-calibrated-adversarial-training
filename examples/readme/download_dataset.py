import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')

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
