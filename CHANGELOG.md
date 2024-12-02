# Changelog

Note: This is the changelog of the C++ library. The Python port has a separate
Changelog under `yggdrasil_decision_forests/port/python/CHANGELOG.md`.

## HEAD

### Features

-   Speed-up training of GBT models by ~10%.
-   Support for categorical and boolean features in Isolation Forests.
-   Rename LAMBDA_MART_NDCG5 to LAMBDA_MART_NDCG. The old name is deprecated but
    can still be used.
-   Allow configuring the truncation of NDCG losses.
-   Add support for distributed training for ranking gradient boosted tree
    models.
-   Add support for AVRO data file using the "avro:" prefix.
-   Deprecated `SparseObliqueSplit.binary_weights` hyperparameter in favor of
    `SparseObliqueSplit.weights`.

### Misc

-   Loss options are now defined
    model/gradient_boosted_trees/gradient_boosted_trees.proto (previously
    learner/gradient_boosted_trees/gradient_boosted_trees.proto)
-   Remove C++14 support.

## 1.10.0 - 2024-08-21

### Features

-   Add support for Isolation Forests model.
-   The default value of `num_candidate_attributes` in the CART learner is
    changed from 0 (Random Forest style sampling) to -1 (no sampling). This is
    the generally accepted logic of CART.
-   Added support for GCS for file I/O.

## 1.9.0 - 2024-03-12

### Feature

-   Add "parallel_trials" parameter in the hyper-parameter tuner to control the
    number of trials to run in parallel.
-   Add support for custom losses.

## 1.8.0 - 2023-11-17

### Feature

-   Support for GBT distances.
-   Remove old snapshots automatically for GBT training.

### Fix

-   Regression with Mean Squared Error loss and Mean Average error loss
    incorrectly clamped the gradients, leading to incorrect predictions.
-   Change dependency from boost to boost_math for faster builds.

## 1.7.0 - 2023-10-20

### Feature

-   Add support for Mean average error (MAE) loss for GBT.
-   Add pairwise distance between examples.
-   By default, only keep the last three snapshots when training with a working
    cache to be resilient to training interruptions.

### New interface

-   Check out the new Python interface in port/python! It's still experimental
    but you can already install it from PyPi with `pip install ydf`.

## 1.6.0 - 2023-09-28

### Breaking changes

-   The dependency to the distributed gradient boosted trees learner is renamed
    from
    `//third_party/yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees`
    to
    `//third_party/yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees:dgbt`.
    Note most case, importing the learners with
    `//third_party/yggdrasil_decision_forests/learner:all_learners` is
    recommended.
-   The training configuration must contain a label. A missing label is no
    longer interpreted as the label being the input feature "".

### Feature

-   Add support for monotonic constraints for gradient boosted trees.
-   Improve speed of dataset reading and writing.

### Fix

-   Proper error message when using distributed training on more than 2^31
    (i.e., ~2B) examples while compiling YDF with 32-bits example index.
-   Fix Window compilation with Visual Studio 2019
-   Improved error messages for invalid training configuration
-   Replaced outdated dependencies

## 1.5.0 - 2023-07-03

### Feature

-   Rename experimental_analyze_model_and_dataset to analyze_model_and_dataset
-   Add new GBT loss function `POISSON` for Poisson log likelihood.
-   Go API: Categorical string values available for inspection.
-   Improved training speed for unit-weight datasets.
-   Support for MHLD oblique decision trees.
-   Multi-threaded RMSE computation.
-   Added Uint8 inference engine.
-   Added Multi-task learning where the output of models trained as "secondary"
    are used as input for the models trained as "primary"

### Fix

-   Go API: fixed typo on OutOfVocabulary constant.
-   Error messages for Uplift models.
-   Remove owner leakage in the model compiler.
-   Fix buggy restriction for SelGB sampling
-   Improve documentation.

### Change

## 1.4.0 - 2023-03-20

### Features

-   Speed-up the computation of PDP and CEP in the model analysis tool.
-   Add compilation of model into .h file.
-   [JS port] Add "prefix" argument to model loading method.
-   Rename logging function from LOG to YDF_LOG to limit risk of collision with
    TF or Absl.

### Fix

-   [JS port] Fix memory leak. Release emscripten objects.

## 1.3.0 - 2023-01-24

### Features

-   Setting the generic hyper-parameter "subsample" is enough enable random
    subsampling (to need to also set "sampling_method=RANDOM").
-   Improve the display of decision tree structures.
-   The Hyper-parameter optimizer field "predefined_search_space" automatically
    configures the set of hyper-parameters to explore during automatic
    hyper-parameter tuning.
-   Replaces the MEAN_MIN_DEPTH variable importance with INV_MEAN_MIN_DEPTH.

## 1.2.0 - 2022-11-18

### Features

-   YDF can load TF-DF models directly (i.e. a TF model with a YDF model in the
    "assets" sub directory).
-   Expose confusion tables in a GBT model's analysis.
-   Add the "compute_variable_importances" tool to compute variable importances
    on an already trained model.
-   Add the "experimental_analyze_model_and_dataset" tool to understand/analyze
    models.

## 1.1.0 - 2022-10-21

### Features

-   Early stopping is no longer triggered during first iterations. The initial
    iteration for early stopping can be controlled with the new parameter
    `early_stopping_initial_iteration` in `gradient_boosted_trees.proto`.
-   Benchmark inference tool does not require for the dataset to contain the
    label column.
-   The user can specify the location of the wasm file in the JavaScript port.
-   The user can instruct the tokenizer to perform no tokenization at all.

### Cleanup

-   Fix GRPC dependency to version 1.50.0.

### Documentation

-   The new documentation is live at
    [ydf.readthedocs.io](https://ydf.readthedocs.io).

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
