# Early Stopping

Yggdrasil's Gradient Boosted Tree learners use **early stopping** to prevent
[overfitting](https://developers.google.com/machine-learning/glossary#overfitting).
This page offers an overview of the configuration of early stopping and
tradeoffs to consider.

At a high level, early stopping is a common ML technique to evaluate a model's current performance against a valuation dataset during training. In the case of
GBTs, the learner adds one (or multiple) trees with each iteration until the
model contains the requested number of trees. With early stopping, the learner
periodically (e.g., at the end of each iteration) computes a *validation loss*
at the end of each iteration, which is the model's loss on the validation
dataset. Based on the validation loss, the learner may decide to include fewer
trees than requested in the final model[^1].

## Early stopping modes

Yggdrasil's early stopping is configured through the
[gradient_boosted_trees.proto](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto).
Yggdrasil supports three modes for early stopping, configured through the
`early_stopping` option:

-   `NONE`: No early stopping is used. \
    If the dataset is very small or if overfitting is unlikely, disabling early
    stopping is a sensible choice.
-   `VALIDATION_LOSS_INCREASE`: Stops the training when the validation
    loss stops decreasing. \
    More precisely, and to account for potential training noise (the validation
    loss can decrease, increase and then decrease again), training stops if the
    smallest (best) validation loss was observed more than
    `early_stopping_num_trees_look_ahead` trees/ training iterations ago.
-   `MIN_VALIDATION_LOSS_ON_FULL_MODEL`: Trains all the "num_trees", and then
    select the subset $[1,..,k]$ of trees that minimizes the validation loss. \
    This solution is can lead to better models than `VALIDATION_LOSS_INCREASE`,
    but it is more expensive as all the trees are always trained

## Additional options

### Lookahead

In mode `VALIDATION_LOSS_INCREASE`, the learner stops training when it is
confident that additional trees will no longer decrease the validation loss.
Parameter `early_stopping_num_trees_look_ahead` controls how many trees the
learner will compute to see if they reduce the validation loss.

The default value is 30. Parameter `early_stopping_num_trees_look_ahead` should
be increased if the learner is suspected to trigger early stopping prematurely,
but it mildly increases the running time of the algorithm and results in larger
models

### Start iteration

Sometimes, the first few training iterations of training is noisy, and the
validation loss oscillates. To avoid triggering it by accident, early stopping
is disabled during the first `early_stopping_initial_iteration` iterations.

During the first few iterations of the learner, the model can still be very
noisy, with the validation loss making surprising jumps. Parameter
`early_stopping_initial_iteration` controls how many of the first iterations
will be skipped when computing the validation loss.

The default value is 1. Parameter `early_stopping_initial_iteration` should be
increased if the learner is suspected to trigger early stopping prematurely, but
it mildly increases the running time of the algorithm and results in larger
models.

### Validation dataset

The validation dataset is picked uniformly at random from the training
dataset. The proportion of the validation dataset is controlled with the parameter
`validation_set_ratio`. The default value is 0.1 that is, the learner will use
10% of the input dataset to form the validation dataset. For small or highly
imbalanced datasets, the validation dataset should be small or early stopping
disabled.

The split between training and validation datasets can be controlled more
precisely with `validation_set_group_feature`. The value of this parameter
should be the name of a feature or a regular expression resolving to one or more
feature names. Two examples with the same feature value(s) as specified by
`validation_set_group_feature` cannot be on different sides of the
training/validation split. This is useful to ensure that highly correlated
examples always land on the same side of the split.

### Validation interval

The learner evaluates the model on the validation set at every
`validation_interval_in_trees` trees. Increasing this value will improve the
algorithm's speed and might lead to slightly worse / larger models.

[^1]: Note that, despite its name, early stopping may still allow the learner
    compute all requested trees and only prune the model afterward.
