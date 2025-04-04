# How to train a model faster?

Some hyper-parameters can have a significant impact on both the model quality
and the training speed.

## Approximated splits

By default, use learning with **exact splits**. The alternative to exact splits
is **approximated splits**, which are much faster (2x to 5x speed-up depending
on the dataset) to learn but can sometimes lead to a drop in quality.

For non-distributed training, enable approximated splits with
`discretize_numerical_columns=True`. The `num_discretized_numerical_bins`
parameter (default to 255) controls the number of bins used for discretizing
numerical columns.

For distributed training, enable approximated splits with
`force_numerical_discretization=True`. The
`max_unique_values_for_discretized_numerical` (default to 16000) parameter
controls the accuracy of the approximate spits. A smaller value will make the
algorithm faster, but it may also result in a less accurate spit.

If training time is limited, using approximate splitting and while optimizing
other hyperparameters can result in both faster training and improved accuracy.

**About other libraries**

In XGBoost, approximated splits can be enabled with `tree_method="hist"`.

LightGBM always uses approximated splits.

## Distributed training

Distributed training divides the computation cost of training a model over
multiple computers. In other words, instead of training a model on a single
machine, the model is trained on multiple machines in parallel. This can
significantly speed up the training process, as well as allow for larger
datasets to be used. On small datasets, distributed training does not help.

## Number of trees

The training time is directly proportional to the number of trees. Decreasing
the number of trees will reduce the training time.

## Candidate attribute ratio

Training time is is directly proportional to the
`num_candidate_attributes_ratio`. Decreasing `num_candidate_attributes_ratio`
will reduce the training time.

## Disable OOB performances [RF only]

When `compute_oob_performances=True` (default), the Out-of-bag evaluation is
computed during training. OOB evaluation is a great way to measure the quality
of a model, but it does not impact training. Disabling
`compute_oob_performances` will speed up Random Forest model training.

## Set a maximum training time

`maximum_training_duration_seconds` controls the maximum training time of a
model.

## Reduce tested oblique projections

When training a sparse oblique model (`split_axis=SPARSE_OBLIQUE`), the number
of tested projection is defined by `num_features^num_projections_exponent`.
Reducing `num_projections_exponent` will speed-up training.

## Increase number of training threads

The default number of training threads is set to the number of cores on the
machine, up to 32. If the machine has more than 32 cores, the number of training
threads is limited to 32. In this case, manually setting the `num_threads`
parameter to a larger number can speed up training.

## Increase shrinkage [GBT only]

The "shrinkage", sometimes referred to as the "learning rate", determines how
quickly a GBT model learns. Learning too quickly typically results in inferior
results but produces smaller, faster-to-train, and faster-to-run models.
`shrinkage` defaults to 0.1. You can try 0.15 or event 0.2.
