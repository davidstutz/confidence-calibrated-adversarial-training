# Training

Training follows the interface defined in `common.train.TrainingInterface` and is split
into the following classes:

* `common.train.NormalTraining`: normal training;
* `common.train.AdversarialTraining`: training on adversarial examples, supporting full adversarial training
and adversarial training on clean _and_ adversarial examples;
* `common.train.ConfidenceCalibratedAdversarialTraining`: confidence-calibrated training on adversarial examples;

Training is done by providing:

* `trainset`: the training set as `torch.utils.data.DataLoader`;
* `testset`: the test set as `torch.utils.data.DataLoader`;
* `optimizer`: an optimizer as from `torch.optim`;
* `scheduler`: a learning rate scheduler from `torch.optim.lr_scheduler`;
* `augmentation`: an _optional_ augmenter, using `imgaug`;