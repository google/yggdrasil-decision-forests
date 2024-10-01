# How to improve a model?

This document provides advice on how to improve the quality, speed, and size of
a YDF model. The amount of improvement will vary depending on the dataset. In
some cases, the changes will be minor, while in others, they may be significant.
It is not possible to know in advance how much improvement a given change will
produce.

This guide is divided in two chapters: **Optimizing model quality** and
**optimizing model speed**. In most cases, improving the model quality will also
make it larger and slower, and vice versa. In other words, the predictive
quality of a model is generally tied to its size.

Having a basic understanding of how decision forests work is useful to optimize
them. For more information, please refer to
[Google's Decision Forests class](https://developers.google.com/machine-learning/decision-forests).

The [hyper-parameter page](hyperparameters.md) lists and explains the available
hyper-parameters.

## Random Forest or Gradient Boosted Trees?

**Random Forests** (RF) and **Gradient Boosted Trees** (GBT) are two different
algorithms for training decision forests. Each algorithm has its own set of
strengths and weaknesses. At a high level, RFs are less prone to overfitting
than GBTs, making them a good choice for small datasets and datasets with a
large number of input features. On the other hand, GBTs learn more efficiently
than RFs. Additionally, GBT models are often much smaller and allow for faster
inference than comparable RF models.

When optimizing for speed, use GBT. When optimizing for quality, both algorithms
should be tested.

**Warning:** Both algorithms have hyperparameters in common, such as the number
of trees and the maximum tree depth. However, these hyperparameters play a
different role in each algorithm and should be tuned accordingly. For example,
the maximum tree depth of a GBT is typically between 3 and 8, while it is rarely
less than 16 in an RF.

## Optimizing model quality

### Automated hyper-parameter tuning

Automated hyperparameter tuning is a simple but expensive solution to improve
the quality of a model. When full hyper-parameter tuning is too expensive,
combining hyper-parameter tuning and manual tuning is a good solution.

See the [Tuning notebook](tutorial/tuning.ipynb) for details.

## Hyper-parameter templates

The default hyperparameters of YDF learners are set to reproduce the originally
published algorithms,a new methods are always disabled by default.

As such, default parameters are not optimized for performance, which can lead to
reasonable, but not optimal, results. To benefit from the latest YDF algorithm
without having understood those hyper-parameters and without having to run
hyper-parameter tuning, YDF have pre-configured **hyper-parameter templates**.

The hyper-parameter templates are available by calling
[hyperparameter_templates](py_api/GradientBoostedTreesLearner.md#ydf.GradientBoostedTreesLearner.hyperparameter_templates)
on a learner.

```python
# List the available templates for the GBT learner.
templates = ydf.GradientBoostedTreesLearner.hyperparameter_templates()
print(templates)

# Use the "better_defaultv1" template:
learner = ydf.GradientBoostedTreesLearner(**templates["better_defaultv1"], ...)
```

Hyper-parameter templates are also available on the
[Hyper-parameter page](hyperparameters/#hyper-parameter-templates). Note that
different learners have different templates.

### Increase the number of trees

The `num_trees` parameter controls the number of trees in the model. Increasing
the number of trees often improve the quality of the model. By default, YDF
trains models with 300 trees. For high quality models, using 1000 or even more
trees is sometimes valuable.

**Note:** When training a gradient boosted trees model with early stopping (the
default behavior), early stopping may reduce the number of trees in the model to
a value less than "num_trees".

### Use oblique trees

By default, trees are "orthogonal" or "axis aligned", that is, each
split/condition tests a single feature. By opposition, conditions in oblique
trees can use multiple features. Oblique splits generally improve performances
by are slower to train.

Oblique trees are more expensive to train. The `num_projections_exponent`
parameter plays an important role in the training time and final model quality
(1 is cheap, 2 is better but more expensive). See `SparseObliqueSplit` in the
[DecisionTreeTrainingConfig](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto#L152)
for more details.

```python
learner = ydf.RandomForestLearner(
  split_axis="SPARSE_OBLIQUE",
  sparse_oblique_normalization="MIN_MAX",
  sparse_oblique_num_projections_exponent=1.0,
  ...)
```

## Random Categorical splits (GBT and RF)

By default, categorical splits are learned with the CART categorical algorithm.
The Random categorical algorithm is another solution that can improve the model
performances at the expense of model size.

```python
learner = ydf.RandomForestLearner(categorical_algorithm="RANDOM", ...)
```

## Reduce shrinkage [GBT only]

The "shrinkage", sometimes referred to as the "learning rate", determines how
quickly a GBT model learns. Learning slowly can improve the model quality.
`shrinkage` defaults to 0.1. You can try 0.05 or 0.02.

## Other impactful hyper-parameters for GBT

While all hyperparameters and can improve the model's quality, some
hyperparameters have a greater impact than others. In addition to the parameters
mentioned above, the following are the most important parameters for GBT:

-   `use_hessian_gain` (default `False`). For example try
    `use_hessian_gain=True`.
-   `max_depth` (default `6`). For example try `max_depth=5`.
-   `num_candidate_attributes_ratio` (default `1`). For example try
    `num_candidate_attributes_ratio=0.9`.
-   `min_examples` (default `5`). For example try `min_examples=10`.
-   `growing_strategy` (default `"LOCAL"`). For example try
    `growing_strategy="BEST_FIRST_GLOBAL"`.

**Note:** When training a model with `growing_strategy=LOCAL` (default), it is
often beneficial to tune the `max_depth` parameter (default 6). When training a
model with `growing_strategy=BEST_FIRST_GLOBAL`, it is best to leave `max_depth`
unconstrained (default -1) and tune the `max_num_nodes` parameter instead.

## Disabling the validation dataset (GBT only)

By default, if not validation dataset is provided, the Gradient Boosted Trees
learner extracts 10% of the training dataset to build a validation dataset to
control early stopping (i.e. stop the training when the model start to overfit).

For both small datasets and large datasets, it might be good to use all the data
for training (and therefore disable early-stopping). In this case, the
`num_trees` parameter should be tuned.

```python
learner = ydf.GradientBoostedTreesLearner(validation_ratio=0.0, ...)
```

**Warning:** Disabling early stopping may cause the model to overfit. To avoid
this, first run your training **with early stopping** to determine the optimal
number of trees. For instance, if early stopping never triggers before the end
of training, you can probably disable it (and use the extra data for training).
If early stopping always triggers close to a given number of trees, you might
also do the same. Keep in mind that changing any other hyperparameter will
require you to retest the behavior of early stopping.

## Optimizing model speed (and size)

The speed and size of a model is constrained by the number of input features,
number of trees and average depth of the trees.

You can measure the inference speed of a model with the
[benchmark](py_api/GradientBoostedTreesModel/#ydf.GradientBoostedTreesModel.benchmark)
method.

```python
model.benchmark(dataset)
```

**Example of results**

```
Inference time per example and per cpu core: 0.702 us (microseconds)
Estimated over 441 runs over 3.026 seconds.
* Measured with the C++ serving API. Check model.to_cpp() for details.
```

### Switch from a Random Forest to a Gradient Boosted Trees

Random Forest models are much larger and slower than Gradient Boosted trees.
When speed is important, use Gradient Boosted trees models.

```python
# Before
learner = ydf.RandomForestLearner(...)

# After
learner = ydf.GradientBoostedTreesLearner(...)
```

### Reduce the number of trees

The `num_trees` parameter controls the number of trees in the model. Reducing
this parameter will decrease the size of the model at the expense of the model
quality.

**Note:** When training a gradient boosted trees model with early stopping (the
default behavior), early stopping may reduce the number of trees in the model to
a value less than "num_trees".

When training with a `growing_strategy="BEST_FIRST_GLOBAL"`, it is best to not
limit the maximum number of trees and to optimize `max_num_nodes` instead.

### Remove model debugging data

YDF models include metadata for model interpretation and debugging. This
metadata is not used for model inference and can be discarded to reduce the
model size. Removing this data will typically reduce the model size by ~50%.
Removing this data does not improve the model's speed.

To train a model without metadata, set the learner constructor argument
`pure_serving_model=True`.

```python
learner = ydf.GradientBoostedTreesLearner(pure_serving_model=True, ...)
```

If using the CLI API, the meta-data can be removed with the `edit_model` CLI
tool:

```shell
# Remove the meta-data from the model
./edit_model --input=/tmp/model_with_metadata --output=/tmp/model_without_metadata --pure_serving=true

# Look at the size of the model
du -h /tmp/model_with_metadata
du -h /tmp/model_without_metadata
```

### Set `winner_take_all_inference=False` with Random Forests

The `winner_take_all_inference` parameter of the Random Forest learner is set to
True by default. This ensures that by default, the YDF Random Forest is
equivalent to the original random forest by Breiman.

However, in many cases `winner_take_all=False` can reduce the size and
improve the quality of a Random Forest model.

```python
learner = ydf.RandomForestLearner(winner_take_all=False, ...)
```

### Set `maximum_model_size_in_memory_in_bytes=...`

The `maximum_model_size_in_memory_in_bytes` parameter controls the maximize size
of the model in RAM. By setting this value, you can control the final size of
the model.

The size of the model in RAM can be larger than the size of the model on disk.
The RAM used to load the model corresponds to the size of the model in RAM.
Before running model inference, the model is compiled into a generally smaller
format.

```python
# Model limited to 10GB
learner = ydf.RandomForestLearner(maximum_model_size_in_memory_in_bytes=10e+9, ...)
```

Different learning algorithms enforce the maximum size differently.

## Increase shrinkage [GBT only]

The "shrinkage", sometimes referred to as the "learning rate", determines how
quickly a GBT model learns. Learning too quickly typically results in inferior
results but produces smaller, faster-to-train, and faster-to-run models.
`shrinkage` defaults to 0.1. You can try 0.15 or event 0.2.
