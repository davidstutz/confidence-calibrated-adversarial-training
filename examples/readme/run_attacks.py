import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')

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