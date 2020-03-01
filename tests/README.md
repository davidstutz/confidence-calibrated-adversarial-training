# Tests

This directory includes unit tests for attacks, training and utilities.
The examples can also be used as examples of how to use or not use the provided code.

Some highlights:

* `test_eval.py`: tests for clean, adversarial, distal and corrupted evaluation, including
confidence-thresholded metrics;
* `test_train.py`: tests for training various models;
* `test_attacks_normal.py` and `test_attacks_adversarial.py`: tests for attacs against undefended and
robust models;