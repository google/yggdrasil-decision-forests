# Changelog

## 1.0.0 - 2022-09-07

### Features

-   Go (GoLang) inference API (Beta): simple engine written in Go to do
    inference on YDF and TF-DF models.
-   Creation of html evaluation report with plots (e.g., ROC, PR-ROC).
-   Add support for Random Forest, CART, regressive GBT and Ranking GBT models
    in the Go API.
-   Add customization of the number of IO threads in the deployment proto.

## 0.2.5 - 2022-06-15

### Features

-   Multithreading of the oblique splitter for gradient boosted tree models.
-   Support for Javascript + WebAssembly inference of model.
-   Support for pure serving model i.e. model containing only serving data.
-   Add "edit_model" cli tool.

### Fix

-   Remove bias toward low outcome in uplift modeling.

## 0.2.4 - 2022-05-17

### Features

-   Discard hessian splits with score lower than the parents. This change has
    little effect on the model quality, but it can reduce its size.
-   Add internal flag `hessian_split_score_subtract_parent` to subtract the
    parent score in the computation of an hessian split score.
-   Add the hyper-parameter optimizer as one of the meta-learner.
-   The Random Forest and CART learners support the `NUMERICAL_UPLIFT` task.

## 0.2.3 - 2021-01-27

### Features

-   Honest Random Forests (also work with Gradient Boosted Tree and CART).
-   Can train Random Forests with example sampling without replacement.
-   Add support for Focal Loss in Gradient Boosted Tree learner.

### Fixes

-   Incorrect default evaluation of categorical split with uplift tasks. This
    was making uplift models with missing categorical values perform worst, and
    made the inference of uplift model possibly slower.

## 0.2.2 - 2021-12-13

### Features

-   The CART learner exports the number of pruned nodes in the output model
    meta-data. Note: The CART learner outputs a Random Forest model with a
    single tree.
-   The Random Forest and CART learners support the `CATEGORICAL_UPLIFT` task.
-   Add `SetLoggingLevel` to control the amount of logging.

### Fixes

-   Fix tree pruning in the CART learner for regressive tasks.

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
