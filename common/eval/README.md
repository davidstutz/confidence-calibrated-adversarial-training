# Evaluation

Evaluation is split into the following classes:

* `common.eval.CleanEvaluation`: evaluation on clean examples;
* `common.eval.AdversarialEvaluation`: evaluation on adversarial examples (and clean examples);
* `common.eval.CorruptedEvaluation`: evaluation on corrupted examples (and clean examples);
* `common.eval.DistalEvaluation`: evaluation on distal adversarial examples;

All cases include common metrics, such as test error, robust test error, area under the
receiver operating characteristic curve (ROC AUC), false positive rate and corrupted test error.
All of these metrics can also be applied in a confidence-thresholded settings.

For further documentation, see the corresponding source code.