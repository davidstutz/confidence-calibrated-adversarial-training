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
