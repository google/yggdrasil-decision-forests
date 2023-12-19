# API Reference

This page documents the Python API for YDF. Users can also train models using
the C++ and CLI APIs.

## Learners

A **Learner** trains models and can be cross-validated.

-   [GradientBoostedTreesLearner](GradientBoostedTreesLearner.md)
-   [RandomForestLearner](RandomForestLearner.md)
-   [CartLearner](CartLearner.md)
-   [DistributedGradientBoostedTreesLearner](DistributedGradientBoostedTreesLearner.md)

All learners derive from [GenericLearner](GenericLearner.md).

## Models

A **Model** makes predictions and can be evaluated.

**Note:** Models (e.g., `GradientBoostedTreesModel`) do not contain training
capabilities. To train a model, you need to create a learner (e.g.,
`GradientBoostedTreesLearner`). Training hyperparameters are constructor
arguments of learner classes.

-   [GradientBoostedTreesModel](GradientBoostedTreesModel.md)
-   [RandomForestModel](RandomForestModel.md)
-   [CARTModel](RandomForestModel.md): Alias to
    [RandomForestModel](RandomForestModel.md).

All models derive from [GenericModel](GenericModel.md).

## Tuners

A **Tuner** finds the optimal set of hyper-parameters using repeated training
and evaluation.

-   [RandomSearchTuner](RandomSearchTuner.md)

## Utilities

-   [verbose](utilities.md#ydf.verbose): Control the amount of logging.
-   [load_model](utilities.md#ydf.load_model): Load a model from disk.
-   [Feature](utilities.md#ydf.Feature): Input feature specific hyper-parameters
    e.g. semantic, constraints.
-   [Column](utilities.md#ydf.Column): Alias for `Feature`.
-   [Task](utilities.md#ydf.Task): Specify the task solved by the model e.g.
    classification.
-   [Semantic](utilities.md#ydf.Semantic): How an input feature in interpreted
    e.g. numerical, categorical.
-   [start_worker](utilities.md#ydf.start_worker): Start a worker for
    distributed training.
-   [strict](utilities.md#ydf.strict): Show more logs.

## Advanced Utilities

-   [ModelIOOptions](utilities.md#ydf.ModelIOOptions):Options to save a model to
    disk.
-   [create_vertical_dataset](utilities.md#ydf.create_vertical_dataset): Load a
    dataset in memory.
-   [ModelMetadata](utilities.md#ydf.ModelMetadata): Meta-data about the model
    e.g. training date, uid.

## Tree

The `ydf.tree.*` classes provides programmatic read and write access to the tree
structure, leaves, and values.

-   [tree.Tree](tree.md#ydf.tree.Tree): A decision tree as returned and consumed
    by `model.get_tree(...)` and `model.set_tree(...)`..

### Conditions

-   [tree.AbstractCondition](tree.md#ydf.tree.AbstractCondition): Base condition
    class.
-   [tree.NumericalHigherThanCondition](tree.md#ydf.tree.NumericalHigherThanCondition):
    Condition of the form `attribute >= threshold`.
-   [tree.CategoricalIsInCondition](tree.md#ydf.tree.CategoricalIsInCondition):
    Condition of the form `attribute in mask`.
-   [tree.CategoricalSetContainsCondition](tree.md#ydf.tree.CategoricalSetContainsCondition):
    Condition of the form `attribute intersect mask != empty`.
-   [tree.DiscretizedNumericalHigherThanCondition](tree.md#ydf.tree.DiscretizedNumericalHigherThanCondition):
    Condition of the form `attribute >= bounds[threshold]`.
-   [tree.IsMissingInCondition](tree.md#ydf.tree.IsMissingInCondition):
    Condition of the form `attribute is missing`.
-   [tree.IsTrueCondition](tree.md#ydf.tree.IsTrueCondition): Condition of the
    form `attribute is true`.
-   [tree.NumericalSparseObliqueCondition](tree.md#ydf.tree.NumericalSparseObliqueCondition):
    Condition of the form `sum(attributes[i] * weights[i]) >= threshold`.

### Nodes

-   [tree.AbstractNode](tree.md#ydf.tree.AbstractNode): Base node class.
-   [tree.Leaf](tree.md#ydf.tree.Leaf): A leaf node containing a value.
-   [tree.NonLeaf](tree.md#ydf.tree.NonLeaf): A non-leaf node containing a
    condition.

### Values

-   [tree.AbstractValue](tree.md#ydf.tree.AbstractValue): Base value class.
-   [tree.ProbabilityValue](tree.md#ydf.tree.ProbabilityValue): A probability
    distribution value.
-   [tree.Leaf](tree.md#ydf.tree.Leaf): The regression value of a regressive
    tree.
-   [tree.UpliftValue](tree.md#ydf.tree.UpliftValue): The uplift value of a
    classification or regression uplift tree.
