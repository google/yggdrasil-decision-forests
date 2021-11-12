# Changelog

## 0.2.1 - 2021-11-05

### Features

-   Add example of distributed training at examples/distributed_training.sh
-   Use the median bucket split value strategy in the discretized numerical
    splitters (local and distributed).

### Fixes

-   Register the GRPC distribution strategy in :train.

## 0.2.0 - 2021-10-29

### Features

-   Distributed training of Gradient Boosted Decision Trees.
-   Add `maximum_model_size_in_memory_in_bytes` hyper-parameter to limit the
    size of the model in memory.

### Fixes

-   Fix invalid splitting of pre-sorted numerical features (make use to use
    midpoint).

## 0.1.5 - 2021-08-11

### Fixes

-   Fix incorrect handling of CART pruning when validation set is empty.
    Previously, the whole tree would be erroneously pruned. Now, pruning is
    disabled if the validation set is not specified.

## 0.1.4 - ????

### Features

-   Add training interruption in the abstract learner API.
-   Reduce the memory usage of the pre-sorted feature index.
-   Multi-threaded computation of the pre-sorted feature index.
-   Disable GBT's early stopping if the validation dataset ratio is zero.
-   Pre-computed and cache the structural variable importances.

## 0.1.3 - 2021-05-19

### Features

-   Register new inference engines.

## 0.1.2 - 2021-05-18

### Features

-   Inference engines: QuickScorer Extended and Pred

## 0.1.1 - 2021-05-17

### Features

-   Migration to TensorFlow 2.5.0.

## 0.1.0 - 2021-05-11

Initial release of Yggdrasil Decision Forests.

### Features

-   CLI: train show_model show_dataspec predict infer_dataspec evaluate
    convert_dataset benchmark_inference utils/synthetic_dataset)
-   Learners: Gradient Boosted Trees (and derivatives), Random Forest (and
    derivatives), Cart.
