# API Reference

This page documents the Python API for YDF. Users can also train models using
the C++ and CLI APIs.

## Learners

A **Learner** trains models and can be cross-validated.

-   [GradientBoostedTreesLearner](GradientBoostedTreesLearner.md)
-   [RandomForestLearner](RandomForestLearner.md)
-   [CartLearner](CartLearner.md)
-   [DistributedGradientBoostedTreesLearner](DistributedGradientBoostedTreesLearner.md)

## Models

A **Model** makes predictions and can be evaluated.

**Note:** Models (e.g., `GradientBoostedTreesModel`) do not contain training
capabilities. To train a model, you need to create a learner (e.g.,
`GradientBoostedTreesLearner`). Training hyperparameters are constructor
arguments of learner classes.

-   [GradientBoostedTreesModel](GradientBoostedTreesModel.md)
-   [RandomForestModel](RandomForestModel.md)

## Tuners

A **Tuner** finds the optimal set of hyper-parameters using repeated training
and evaluation.

-   [RandomSearchTuner](RandomSearchTuner.md)

## Utilities

-   [verbose](utilities.md#verbose): Control the amount of logging.
-   [load_model](utilities.md#load_model): Load a model from disk.
-   [Feature](utilities.md#Feature): Input feature specific hyper-parameters
    e.g. semantic, constraints.
-   [Column](utilities.md#Column): Alias for `Feature`.
-   [Task](utilities.md#Task): Specify the task solved by the model e.g.
    classification.
-   [Semantic](utilities.md#Semantic): How an input feature in interpreted e.g.
    numerical, categorical.
-   [start_worker](utilities.md#start_worker): Start a worker for distributed
    training.
-   [strict](utilities.md#strict): Show more logs.

## Advanced Utilities

-   [ModelIOOptions](utilities.md#ModelIOOptions):Options to save a model to
    disk.
-   [create_vertical_dataset](utilities.md#create_vertical_dataset): Load a
    dataset in memory.
