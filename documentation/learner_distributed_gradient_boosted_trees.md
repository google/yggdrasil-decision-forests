# Distributed Gradient Boosted Trees

## Table of Contents

<!--ts-->

*   [Distributed Gradient Boosted Trees](#distributed-gradient-boosted-trees)
    *   [Table of Contents](#table-of-contents)
    *   [Introduction](#introduction)
    *   [Features](#features)
    *   [Training speed](#training-speed)
    *   [Distribution parameter tuning](#distribution-parameter-tuning)
        *   [RAM](#ram)
        *   [RAM IO (memory bandwidth)](#ram-io-memory-bandwidth)
        *   [CPU](#cpu)
        *   [IO](#io)
    *   [Limitations](#limitations)

<!--te-->

## Introduction

The `DISTRIBUTED_GRADIENT_BOOSTED_TREES` learner is an exact distributed
implementation of the GRADIENT_BOOSTED_TREES learner. This algorithm is an
extension of the
[Exact Distributed Training: Random Forest with Billions of Examples](https://arxiv.org/abs/1804.06755)
algorithm to Gradient Boosted Trees models.

## Features

-   Classification (binary and multi-class) and regression problems.

-   Numerical, boolean and categorical features.

-   Worker and manager interruption.

## Training speed

The training speed is impacted by:

-   The number of features and the number of training examples.
-   The number of workers.
-   The number of trees: `num_trees`.
-   The number / ratio of candidate attributes at each nodes:
    `num_candidate_attributes_ratio`.
-   The faction of numerical features. Setting
    `force_numerical_discretization=True` can significantly speed-up the
    training speed.

## Distribution parameter tuning

This section discusses how to allocate computation resources for the algorithm.

### RAM

The dataset is divided feature-wise among the workers. Each worker loads a
subset of the features and the labels in memory.

The memory usage of a feature/label is given as follows:

1.  If `force_numerical_discretization` is false (default), numerical features
    with more than `max_unique_values_for_discretized_numerical` (default to
    16k) unique values are stored as float32+variable length integers (i.e. 4-8
    bytes per values). If `force_numerical_discretization` is true, all the
    numerical features are downsampled and treated as if they had less than
    `max_unique_values_for_discretized_numerical` unique values.

1.  Numerical features with less than
    `max_unique_values_for_discretized_numerical` unique values are stored as 2
    byte integers.

1.  Categorical features are stored as variable length integers (i.e. 1-4 bytes
    per values).

For example, for a regression dataset with 1B examples, 1k numerical features
and 100 workers, each worker requires 1B * (10 features/workers * 2 bytes + 4
bytes (labels) + 4 bytes (extra)) = 28 GB of memory. The size of the dataset in
memory is printed by the workers at loading time.

In addition, each worker consumes an extra 32 (default) or 64 bits per example
depending on the YGGDRASIL_EXAMPLE_IDX_32_BITS or YGGDRASIL_EXAMPLE_IDX_64_BITS
compilation option.

Finally, each worker require an extra 1bit per example and per worker.

TODO(gbm): Update doc when the load balancer is available.

### RAM IO (memory bandwidth)

The learning algorithm is bounded by memory bandwidth. Numerical features with
more than `max_unique_values_for_discretized_numerical` unique values are
especially expensive (if `force_numerical_discretization` is false).

### CPU

Each of the `w` workers uses `num_threads` threads (as specified in the
deployment configuration).

During training, each of the `w` workers receive `1/w` of the features. Features
are processed in parallel. In addition, individual features (except the non
discretized numerical features i.e. numerical features with more than
`max_unique_values_for_discretized_numerical` unique values when
`force_numerical_discretization=false`) are also processed in parallel.

For example, in a training on 1000 features, 100 workers and 40 threads per
workers (`num_threads=40`). Each worker will process 10 features, and each
feature will be processed by 4 threads.

Because the algorithm is mostly memory-bandwidth bounded, setting `num_threads`
to the number of available core +10% is a good rule of thumb.

### IO

During the preparation phase, the dataset needs to be indexed. This indexing
process is distributed among the workers. The dataset index is stored in the
`cache_path` directory. The size of the cache is similar to the RAM size
discussed above. Note that if the original dataset is stored as a RecordIO or
TFRecord of TF.Example, the dataset cache can be much smaller (e.g. 5x less)
than the original dataset.

## Limitations

-   Only support training from a dataset path (i.e. training on in-memory
    dataset is not allowed).

-   Only support a subset of the features (e.g. hyper-parameters) proposed in
    GradientBoostedTreesLearner. In this case, an error message will be raised
    during the DistributedGradientBoostedTreesLearner object construction.

-   At the start of the training, the dataset is divided by columns and shards
    and then indexed. For this reason, it is best for the dataset to be sharded.

-   (Currently) Does not support a validation dataset and early stopping.

-   (Currently) The training is limited by the speed of the slowest worker.
