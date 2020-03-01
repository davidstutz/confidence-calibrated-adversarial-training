import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')

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