# Changelog

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

##  0.0.5 - 2023-11-10

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
