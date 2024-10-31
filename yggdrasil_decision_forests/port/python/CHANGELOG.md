# Changelog

## HEAD

### Breaking

-   Classification Label classes are now consistently ordered lexicographically
     (for string labels) or increasingly (for integer labels).
-   Change typo partial_depepence_plot to partial_dependence_plot on
    model.analyze().

### Feature

-   Add support for Avro file for path / distributed training with the "avro:"
    prefix.
-   Add support for discretized numerical features for in-memory datasets.

### Feature

-   Expose MRR for ranking models.
-   Add `model.predict_class` to generate the most likely predicted class of
    classification models.

## 0.8.0 - 2024-09-23

### Breaking

-   Disallow positional parameters for the learners, except for label and task.
-   Remove the unsupported / invalid hyperparameters from the Isolation Forest
    learner.
-   Remove parameters for distributed training and resuming training from
    learners that do not support these capabilities.
-   By default, `model.analyze` for a maximum of 20 seconds (i.e.
    `maximum_duration=20` by default).
-   Convert boolean values in categorical sets to lowercase, matching the
    treatment of categorical features.

### Feature

-   Warn if training on a VerticalDataset and fail if attempting to modify the
    columns in a VerticalDataset during training.
-   User can override the model's task, label or group during evaluation.
-   Add `num_examples_per_tree()` method to Isolation Forest models.
-   Expose the slow engine for debugging predictions and evaluations with
    `use_slow_engine=True`.
-   Speed-up training of GBT models by ~10%.
-   Support for categorical and boolean features in Isolation Forests.
-   Add `ydf.util.read_tf_record` and `ydf.util.write_tf_record` to facilitate
    TF Record datasets usage.
-   Rename LAMBDA_MART_NDCG5 to LAMBDA_MART_NDCG. The old name is deprecated but
    can still be used.
-   Allow configuring the truncation of NDCG losses.
-   Enable multi-threading when using `model.predict` and `model.evaluate`.
-   Default number of threads of `model.analyze` is equal to the number of
    cores.
-   Add multi-threaded results in `model.benchmark`.
-   Add argument to control the maximum duration of `model.analyze`.
-   Add support for Unicode strings, normalize categorical set values in the
    same way as categorical values, and validate their types.
-   Add support for distributed training for ranking gradient boosted tree
    models.

### Fix

-   Fix labels of regression evaluation plots
-   Improved errors if Isolation Forest training fails.

### Release music

Perpetuum Mobile "Ein musikalischer Scherz", Op. 257. Johann Strauss (Sohn)

## 0.7.0 - 2024-08-21

### Feature

-   Expose `validate_hyperparameters()` on the learner.
-   Clarify which parameters in the learner are optional.
-   Add support in JAX FeatureEncoder for non-string categorical feature values.
-   Improve performance of Isolation Forests.
-   Models can be serialized/deserialized to/from bytes with `model.serialize()`
    and `ydf.deserialize_model`.
-   Models can be pickled safely.
-   Native support for Xarray as a dataset format for all operations (e.g.,
    training, evaluation, predictions).
-   The output of `model.to_jax_function` can be converted to a TensorFlow Lite
    model.
-   Change the default number of examples to scan when training on files to
    determine the semantic and dictionaries of columns from 10k to 100k.
-   Various improvements of error messages.
-   Evaluation for Anomaly Detection models.
-   Oblique splits for Anomaly Detection models.
-   Reduce significatively the RAM usage of distributed training with
    discretized numerical values.

### Fix

-   Fix parsing of multidimensional ragged inputs.
-   Fix isolation forest hyperparameter defaults.
-   Fix bug causing distributed training to fail on a sharded dataset containing
    an empty shard.
-   Handle unordered categorical sets in training.
-   Fix dataspec ignoring definitions of unrolled columns, such as
    multidimensional categorical integers.
-   Fix error when defining categorical sets for non-ragged multidimensional
    inputs.
-   MacOS: Fix compatibility with other protobuf-using libraries such as
    Tensorflow.
-   Fix parsing of NAs in Xarray datasets.

#### Release music

Rondo Alla ingharese quasi un capriccio "Die Wut über den verlorenen Groschen",
Op. 129. Ludwig van Beethoven

## 0.6.0 - 2024-07-04

### Feature

-   `model.to_jax_function` now always outputs a FeatureEncoder to help feeding
    data to the JAX model.
-   The default value of `num_candidate_attributes` in the CART learner is
    changed from 0 (Random Forest style sampling) to -1 (no sampling). This is
    the generally accepted logic of CART.
-   `model.to_tensorflow_saved_model` support preprocessing functions which have
    a different signature than the YDF model.
-   Improve error messages when feeding wrong size Numpy arrays.
-   Add option for weighted evaluation in `model.evaluate`.

### Fix

-   Fix display of confusion matrix with floating point weights.

### Known issues

-   MacOS build is broken.

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
