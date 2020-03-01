# Examples

This directory includes the following ready-to-use examples supporting MNIST, SVHN and Cifar10:

* `normal_training_robustness.py`: training a normal, undefended model and evaluate it against a standard PGD
adversarial attack in the `L_inf` norm;
* `adversarial_training_robustness.py`: train a model adversarially on PGD adversarial attacks and evaluate
against _strong_ PGD attacks, with up to 50 attempts and 1000 iterations;
* `confidnece_calibrated_adversarial_training_robustness.py`: use confidence-calibrated adversarial training
to obtain a robust model and evaluat it against _strong_ PGD attacks, including adaptive attacks;

Note that these examples do not include all attacks as reported in the paper.