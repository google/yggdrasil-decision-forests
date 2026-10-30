# Changelog

Note: This is the changelog of the C++ library. The Python port has a separate
Changelog under `yggdrasil_decision_forests/port/python/CHANGELOG.md`.

## 1.13.0 - 2026-10-28

This release introduces an unusually large number of bugfixes. While the
majority of bugfixes concern rarely-used cases, we recommend users carefully
monitor model quality, both for existing models and re-trained models. Most
notably, users of POISSON loss and users of the slow engine may see different
predictions for existing models. Additionally, users of MHLD oblique splits,
`use_hessian_gain=True`, Cox Loss, CART trees, GOSS sampling and honest trees
should expect model changes beyond the movement of the random seed.

### Breaking Changes

-   Removed the deprecated model compiler (`cli/compile_model`). Use the
    standalone C++ export instead.
-   Removed `distribute_cli`.
-   `compute_oob_variable_importances=true` now requires
    `compute_oob_performances=true` instead of silently enabling it.
-   Distributed hyperparameter tuning with cross-validation now fails. This
    option was exposed, but cross-validation was not actually used.
-   Training with all example weights equal to zero now fails.
-   Classification self-evaluation reports loss instead of deviance (values
    halve); no training change.

### Features

-   Support for Model Calibration: Decision Forest binary classification models
    can be calibrated on a calibration dataset using the `postprocessors` field
    of the training config. Model calibration information is also available in
    the model evaluation. Calibrated models can be served as other YDF models,
    including the standalone C++ and Java exports.
-   GBDT models set `early_stopping_triggered` if early stopping triggered.
    Thanks [@muscionig](https://github.com/muscionig)!
-   "True" and "False" are now recognized as boolean in CSV files. This change
    only changes the behaviour of columns explicitly specified as BOOLEAN.
-   Support for guided oblique splits. Note that guided oblique splits are still
    experimental.
-   Add support for custom secondary evaluation metrics in GBT training. Thanks
    [@muscionig](https://github.com/muscionig)!
-   Add support for MSLE metric.
-   Add MSE secondary metric for GBT regression.
-   Java standalone: Add `use_runtime_derived_resource_path`.
-   New hyperparameter `sparse_oblique_include_axis_aligned_splits`.
-   Full evaluation inside the self-evaluation.
-   New hyperparameter `min_sum_hessian_in_leaf`.
-   Allow hyperparameter tuning with monotonicity constraints.
-   Support additional metrics as hyperparameter optimization objectives.
-   New hyperparameters `categorical_random_num_trial_exponent`,
    `categorical_random_max_num_trials`
-   Enable Highway Dynamic Dispatch by default for Serving.
-   Add a "time" field in GBT training logs.
-   Add legends / axis annotations to plots.
-   Training with sparse oblique splits is much faster, up to 3x for Random
    Forests on high-dimensional data
    ([#354](https://github.com/google/yggdrasil-decision-forests/pull/354),
    [#356](https://github.com/google/yggdrasil-decision-forests/pull/356),
    [#403](https://github.com/google/yggdrasil-decision-forests/pull/403)).
    Thanks [@ariellubonja](https://github.com/ariellubonja)!

### Fix

-   Fix out-of-range categorical inference engine crashes.
-   Fix slow engine predictions for oblique splits with missing values,
    `initialize_with_class_priors=True`, ranking with SQUARED_ERROR loss and
    POISSON loss with `output_logits=False`.
-   Fix POISSON loss with `output_logits=False` for the generic engine.
-   Fix multi-class GBT with `output_logits=False` and
    `initialize_with_class_priors=True`.
-   Standalone C++ export: Fix Random Forest SCORE aggregation and IF_ELSE
    floating point precision.
-   Fixed many typos in documentation and error messages.
-   Fix initial predictions of MAE and POISSON losses with zero weights, and
    use a more robust softmax for the multinomial loss.
-   Fix CART pruning. This change broadly improves model quality and decreases
    model size.
-   Fix example-weighted CART pruning.
-   Fix Hessian gain with example weights.
-   Fix MHLD oblique splits LDA computation.
-   Fix Hessian split score computation and clamping, esp under monotonicity
    constraints.
-   Fix honest splits to account for unique examples
    ([#216](https://github.com/google/yggdrasil-decision-forests/pull/216)).
    Thanks [@PSSF23](https://github.com/PSSF23)!
-   Fix GOSS sampling to align with the paper.
-   Fix Cox loss overflows.
-   Fix early stopping edge cases (e.g. DART).
-   Deterministic variable-importance order on ties.
-   Fix PDP/CEP for all-nan numerical features.
-   Ignore Nans in Confidence interval computation.
-   Fix overflow / truncation of num_pos_training_examples_with_weight metadata
    with large or fractional weights.
-   Fix validation of oblique conditions.
-   Cancel pending candidates if HP tuner finishes early.
-   Fix smaller HP tuner issues, e.g. integer rounding for negative
    hyperparameters.
-   Set the accuracy field in the evaluation proto.
-   Remove old checkpoints in distributed GBT learner according to
    `max_kept_snapshots`
-   Distributed training: Fix async validation, model corruption on failed
    checkpoints, early unloading of features and release of discarded gRPC
    stubs on worker relocation.
-   Fix double-sqrt bug in training log RMSE encoding.
-   Various crashes.
-   Various performance improvements.

### Infra

-   Use hermetic Python to build YDF.
-   Update dependencies.

## 1.12.0 - 2026-02-04

### API Changes

-   The `Uplift` task is not currently compatible with the `honest=True`
    hyperparameter. Support for honest uplift trees is not yet implemented.
    Training an uplift model with `honest=True` will now raise an error. Please
    reach out to the YDF team if you have a use case for this functionality.

### Features

-   Allow custom fields in model metadata.
-   Add export to standalone Java.
-   Added the Highway Quickscorer inference engines for faster inference on
    supported machines.
-   Various performance improvements.

### Fix

-   Fix honest trees implementation.
-   Various bugfixes
-   Random Forests training no longer trains recursively, which fixes a number
    of crashes in debug mode.

### Infra

-   YDF and Yggdrasil Decision Forests now use Bazel 7 and Bzlmod.

## 1.11.0 - 2025-03-12

### Features

-   Speed-up training of GBT models by ~10%.
-   Support for categorical and boolean features in Isolation Forests.
-   Rename LAMBDA_MART_NDCG5 to LAMBDA_MART_NDCG. The old name is deprecated but
    can still be used.
-   Allow configuring the truncation of NDCG losses.
-   Add support for distributed training for ranking gradient boosted tree
    models.
-   Add support for NUMERICAL_VECTOR_SEQUENCE features.
-   Add support for AVRO data file using the "avro:" prefix.
-   Additional hyperparameters restricting weights of sparse oblique splits
    to integers or powers of 2.
-   Facilitate training on VertexAI.
-   Deprecated `SparseObliqueSplit.binary_weights` hyperparameter in favor of
    `SparseObliqueSplit.weights`.
-   Add Gzip-compressed BLOB_SEQUENCE serialization
-   Enable Poisson loss for model analysis and fast inference.
-   Add config for compatibility with protobuf lite.

### Fix

-   Fix structural variable importances for oblique splits.
-   Deflake tests.
-   Remove CHECK/FATAL from training code.
-   Fix crash in YDF distributed training.

### Misc

-   Loss options are now defined
    model/gradient_boosted_trees/gradient_boosted_trees.proto (previously
    learner/gradient_boosted_trees/gradient_boosted_trees.proto)
-   Remove C++14 support.
-   Various documentation improvements.

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
