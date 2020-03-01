import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')

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
