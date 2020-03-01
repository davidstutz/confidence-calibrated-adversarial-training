import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')

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
