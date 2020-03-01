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