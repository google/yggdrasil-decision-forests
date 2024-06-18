# Changelog

## 0.5.0 - 2024-06-17

### Feature

-   Add support for Isolation Forests model.
-   Add `max_depth` argument to `model.print_tree`.
-   Add `verbose` argument to `train` method which is equivalent but sometime
    more convenient than`ydf.verbose`.
-   Add SKLearn to YDF model converter: `ydf.from_sklearn`.
-   Improve error messages when calling the model with non supported data.
-   Add support for numpy 2.0.

### Tutorials

-   Add anomaly detection tutorial.
-   Add YDF and JAX model composition tutorial.

### Fix

-   Fix error when plotting oblique trees (`model.plot_tree`) in colab.

## 0.4.3- 2024-05-07

### Feature

-   Add `model.to_jax_function()` function to convert a YDF model into a JAX
    function that can be combined with other JAX operations.
-   Print warnings when categorical features look like numbers.
-   Add support for Python 3.12.

### Fix

-   Fix cross-validation for non-classification learners.
-   Fix missing ydf/model/tree/plotter.js
-   Solve dependency collision of YDF Proto between PYDF and TF-DF.

## 0.4.2- 2024-04-22

### Feature

-   Show error message when TF-DF and YDF shared proto dependencies are
    colliding.
-   Add option to specify a validation dataset for the CartLearner.
-   `DecisionForestsLearner` is an alias to `CartLearner`.

## 0.4.1- 2024-04-18

### Fix

-   Solve dependency collision to YDF between PYDF and TF-DF. If TF-DF is
    installed after PYDF, importing YDF will fails with a `has no attribute
    'DType'` error.
-   Allow for training on cached TensorFlow dataset.

## 0.4.0 - 2024-04-10

### Feature

-   Multi-dimensional features can be selected / configured with the `features=`
    training argument.
-   Programmatic access to partial dependence plots and variable importances.
-   Add `model.to_tensorflow_function()` function to convert a YDF model into a
    TensorFlow function that can be combined with other TensorFlow operations.
    This function is compatible with Keras 2 and Keras 3.
-   Add arguments `servo_api=False` and `feed_example_proto=False` for
    `model.to_tensorflow_function(mode="tf")` to export TensorFlow SavedModel
    following respectively the Servo API and consuming serialized TensorFlow
    Example protos.
-   Add `pre_processing` and `post_processing` arguments to the
    `model.to_tensorflow_function` function to pack pre/post processing
    operations in a TensorFlow SavedModel.

### Tutorials

-   Add tutorial
    [Vertex AI with TF Serving](https://ydf.readthedocs.io/en/latest/tutorial/tf_serving/)
-   Add tutorial
    [Deep-learning with YDF and TensorFlow](https://ydf.readthedocs.io/en/latest/tutorial/compose_with_tf/)

## 0.3.0 - 2024-03-15

### Breaking

-   Custom losses now require to provide the gradient, instead of the negative
    of the gradient.
-   Clarified that YDF may modify numpy arrays returned by a custom loss
    function.

### Features

-   Allow using Jax for custom loss definitions.
-   Allow setting `may_trigger_gc` on custom losses.
-   Add support for MHLD oblique decision trees.
-   Expose hyperparameter `sparse_oblique_max_num_projections`.
-   HTML plots for trees with `model.plot_tree()`.
-   Fix protobuf version to 4.24.3 to fix some incompatibilities when using
    conda.
-   Allow to list compatible engines with `model.list_compatible_engines()`.
-   Allow to choose a fast engine with `model.force_engine(...)`.

### Fix

-   Fix slow engine creation for some combination of oblique splits.
-   Improve error message when feeding multi-dimensional labels.

### Documentation

-   Clarified documentation of hyperparameters for oblique splits.
-   Fix plots, typos.

#### Release music

Doctor Gradus ad Parnassum from "Children's Corner" (L. 113). Claude Debussy

## 0.2.0 - 2024-02-22

### Features

-   Add support for custom losses for GBTs (Classification & Regression).

### Fixes

-   In the tuner, run `parallel_trials` trials in parallel, instead of
    `num_threads` trials in parallel.
-   Force validation dataset as path if training dataset is a path.
-   Fix benchmark.
-   Clarify error when using an incompatible dataset when ranking.
-   Improve test coverage.
-   Fix an issue with the conditions when building trees manually.

### Documentation

-   Add a migration guide TF-DF ==> YDF.
-   Clarify max_depth parameter documentation.
-   Clarify early stopping documentation.

#### Release music

Galop infernal d'Orphée aux Enfers. Jacques Offenbach

## 0.1.0 - 2024-01-25

### Features

-   Added model validation evaluation (for GBTs) and OOB evaluation (for RFs).
-   Expose winner-takes-all for Random Forests.
-   Added model self evaluation.
-   Added `ydf.from_tensorflow_decision_forests()` for importing TF-DF models.
-   Allow feeding datasets as sequence of strings.

### Fixes

-   Fixes a plotting issue for GBTs without validation loss

#### Release music

Flötenuhren von 1772 und 1793 - Vivace (Hob XIX:13). Joseph Haydn

## 0.0.8 - 2023-12-19

### Features

-   Improve training interruption in Jupyter notebooks.
-   Training and inference is compatible with python threading.
-   Add the model inspector, which allows to programmatically inspect the
    structure of trees.
-   Add a model editor, which allows to programmatically modify the structure of
    trees.

### Documentation

-   Many new tutorials at ydf.readthedocs.io

### Bug

-   Fix display of plots in the model analysis.

### Release music

Les Anges dans nos campagnes. Traditional

## 0.0.7 - 2023-11-28

### Features

-   Support for prediction / evaluation for models with N/A conditions.
-   Support for predicting from datasets without label column
-   Model inspector
-   Automatic inference for Categorical set columns

#### Release music

Knecht Ruprecht aus "Album für die Jugend" (Opus 86). Robert Schumann

## 0.0.6 - 2023-11-28

### Features

-   Add support categorical set features
-   Add support for prediction analysis
-   Expose max_num_scanned_rows_to_infer_semantic and
    max_num_scanned_rows_to_compute_statistics
-   Expose metadata on the model
-   Expose initial predictions on the model
-   Add strict mode for more warnings
-   Add GBT example distance support
-   Support strided numerical features

### Documentation

-   Many new tutorials at ydf.readthedocs.io

### Fix

-   Fix a bug in GBT predictions (see YDF changelog for details)
-   Various small fixes

#### Release music

Die Holzauktion. Franz Meißner

## 0.0.5 - 2023-11-10

### Features

-   Add support for distributed training.
-   Allow setting node format.
-   Allow programmatic access to variable importances
-   Support for multi-dimensional numerical features
-   Support use of hyper-parameter templates.

### Documentation

-   Many new tutorials at ydf.readthedocs.io
-   Updated FAQ

### Fix

-   Fix prediction array for multi-class classification.
-   Avoid exposing proto Task type
-   No longer expose train_from_path and train_from_dataset -- just use `train`!
-   Correctly choose number of threads if 32 cores are detected.
-   Logs were hanging in some configurations.

#### Release music

Auld Lang Syne. Traditional

## 0.0.4 - 2023-11-01

### Features

-   New documentation launched!
-   Add support for ranking tasks
-   Add support for reading from path, supporting multiple data types (csv, ...)
-   Improved model descriptions
-   Add support for resuming training
-   Add export to Tensorflow
-   Control logging verbosity
-   Add `model.distance(...)` to compute pairwise distance between examples.
-   Expose data spec on the model
-   Support for monotonic constraints
-   Simple training benchmark

### Fix

-   Remove pybind11_abseil to avoid clashes with other libraries such as
    Tensorflow.

#### Release music

Rhapsody in Blue. George Gershwin

## 0.0.3 - 2023-10-20

### Features

-   Hyperparameter tuning
-   Export to C++
-   Benchmark inference speed
-   Tree leaves retrieval
-   C++ base updated to 1.7.0

#### Release music

Schweigt stille, plaudert nicht (BWV 211). Johann Sebastian Bach

## 0.0.2 - 2023-10-05

### Features

-   Model analysis

#### Release music

When You're Smiling. Larry Shay, Mark Fisher and Joe Goodwin

## 0.0.1 - 2023-10-03

Initial Alpha Release

### Features

-   Training, prediction, evaluation, Import from Pandas
-   Learners: Gradient Boosted Trees (and derivatives), Random Forest (and
    derivatives), Cart.

#### Release music

Frisch heran (Opus 386). Johann Strauss (Sohn)
