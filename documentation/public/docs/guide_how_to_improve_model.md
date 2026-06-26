# How to Improve a Model

This document provides advice on how to improve the quality, speed, and size of
a YDF model. The amount of improvement will vary depending on the dataset. In
some cases, the changes will be minor, while in others, they may be significant.
It is not possible to know in advance how much a given change will improve a
model.

This guide is divided into two sections: **Optimizing Model Quality** and
**Optimizing Model Speed and Size**. In most cases, improving model quality will
also make it larger and slower, and the other way around. In other words, a
model's predictive quality is generally tied to its size and complexity.

Having a basic understanding of how decision forests work is helpful for
optimizing them. For more information, please refer to Google's Decision Forests
class. The [hyper-parameter page](../hyperparameters) also lists and explains
all available options.

## Random Forest or Gradient Boosted Trees?

Random Forests (RF) and Gradient Boosted Trees (GBT) are two different
algorithms for training decision forests, and each has its own strengths.

At a high level, Random Forests are less prone to overfitting than GBTs, making
them a good choice for smaller datasets or datasets with many input features. On
the other hand, Gradient Boosted Trees learn more efficiently. GBT models are
often much smaller and allow for faster inference than comparable RF models.

When optimizing for speed, use GBT.

When optimizing for quality, both algorithms should be tested.

!!! warning

    Both algorithms share some hyperparameter names, such as num_trees and
    max_depth. However, these hyperparameters play a different role in each
    algorithm and must be tuned accordingly. For example, the max_depth of a GBT is
    typically between 3 and 8, while for an RF, it is rarely less than 16.

## Optimizing Model Quality

### Automated Hyperparameter Tuning

Automated hyperparameter tuning is a simple but computationally expensive way to
improve a model's quality. If a full tuning search is too costly, a good
strategy is to combine automated tuning for the most important parameters with
manual tuning for the rest.

See the [Tuning notebook](tutorial/tuning.ipynb) for a practical example.

### Hyperparameter Templates

YDF's default hyperparameters are configured to reproduce the originally
published algorithms, and new methods are always disabled by default. As such,
the defaults produce reasonable but not always optimal results.

To benefit from YDF's latest algorithmic improvements without deep-diving into
every hyperparameter, you can use pre-configured hyperparameter templates.

You can list the available templates by calling
[hyperparameter_templates](../py_api/GradientBoostedTreesLearner/#ydf.GradientBoostedTreesLearner.hyperparameter_templates)
on a learner.

```python
# List the available templates for the GBT learner.
templates = ydf.GradientBoostedTreesLearner.hyperparameter_templates()
for name, params in templates.items():
  print(f"{name}: {params}")

# Use the "better_defaultv1" template:
learner = ydf.GradientBoostedTreesLearner(
    **templates["better_defaultv1"],
    label="my_label"
)
```

### Increase the Number of Trees

The num_trees parameter controls the number of trees in the model. Increasing
this number often improves model quality. By default, YDF trains models with 300
trees, but for high-quality models, using 1000 or even more can be valuable.

!!! note

    When training a GBT model with early stopping (the default behavior), the
    final model may contain fewer trees than specified by num_trees if performance
    on the validation set stops improving.

### Use Oblique Trees

By default, trees are "orthogonal" or "axis-aligned", meaning each split tests a
single feature. In contrast, conditions in oblique trees can test linear
combinations of multiple features. Oblique splits generally improve performance
but are slower to train.

The `num_projections_exponent` parameter plays an important role in the
trade-off between training time and model quality (a value of 1.0 is cheaper,
while 2.0 is better but more expensive).

```python
learner = ydf.RandomForestLearner(
    label="my_label",
    split_axis="SPARSE_OBLIQUE",
    sparse_oblique_normalization="MIN_MAX",
    sparse_oblique_num_projections_exponent=1.5,
)
```

### Use Random Categorical Splits (GBT and RF)

By default, splits on categorical features are learned using the CART algorithm.
The RANDOM algorithm is an alternative that can improve model performance,
sometimes at the expense of a larger model size.

```python
learner = ydf.RandomForestLearner(categorical_algorithm="RANDOM", label="my_label")
```

### Reduce Shrinkage (GBT only)

The shrinkage, also known as the "learning rate", determines how quickly a GBT
model learns. A smaller value forces the model to learn more slowly, which can
improve its final quality. The default shrinkage is 0.1; you can try smaller
values like 0.05 or 0.02.

### Other Impactful Hyperparameters for GBT

While all hyperparameters can impact quality, some are more influential than
others. Besides the ones already mentioned, consider tuning the following for
GBTs:

*   `use_hessian_gain` (default: `False`): Try setting to True.
*   `max_depth` (default: `6`): Try other values, like 5 or 8.
*   `num_candidate_attributes_ratio` (default: `1.0`): Try a value like 0.9.
*   `min_examples` (default: `5`): Try a larger value like `10`.
*   `growing_strategy` (default: `"LOCAL"`): Try `"BEST_FIRST_GLOBAL"`.

!!! note

    When using `growing_strategy="LOCAL"` (the default), it is often beneficial
    to tune `max_depth`. When using `growing_strategy="BEST_FIRST_GLOBAL"`, it is
    better to leave `max_depth` unconstrained (the default value of -1 does this)
    and tune `max_num_nodes` instead.

### Disable the Validation Dataset (GBT only)

By default, the GBT learner reserves 10% of the training data as a validation
dataset for early stopping (i.e., to stop training when the model starts to
overfit).

For very small or very large datasets, it can be beneficial to use all the data
for training by disabling early stopping. In this case, the `num_trees`
parameter should be carefully tuned.

```python
# Disable the validation set and use all data for training
learner = ydf.GradientBoostedTreesLearner(validation_ratio=0.0, label="my_label")
```

!!! warning

    Disabling early stopping can lead to overfitting. Before disabling
    it, train with early stopping enabled and observe when it triggers. If it rarely
    triggers, or always triggers near a specific number of trees, disabling it might
    be safe. Remember that changing other hyperparameters may require you to
    re-evaluate this decision.

## Optimizing Model Speed and Size

The inference speed and size of a model are constrained by the number of input
features, the number of trees, and the average depth of the trees.

You can measure the inference speed of a model with the
[benchmark()](../py_api/GradientBoostedTreesModel/#ydf.GradientBoostedTreesModel.benchmark)
method.

```python
model.benchmark(dataset)
```

### Switch from Random Forest to Gradient Boosted Trees

Random Forest models are typically much larger and slower at inference time than
GBTs. When speed is important, prefer GBT models.

```python
# Before
learner = ydf.RandomForestLearner(...)

# After
learner = ydf.GradientBoostedTreesLearner(...)
```

### Reduce the Number of Trees

The num_trees parameter controls the number of trees in the model. Reducing this
value will decrease the model's size and improve inference speed at the expense
of quality.

### Remove Model Debugging Data

YDF models include metadata for interpretation and debugging (e.g., for
`model.describe()`). This metadata is not used for inference and can be removed
to reduce the final model size, often by as much as 50%. This does not affect
the model's inference speed.

To train a model without this metadata, set `pure_serving_model=True` in the
learner.

```python
learner = ydf.GradientBoostedTreesLearner(pure_serving_model=True, label="my_label")
```

### Set `winner_takes_all=False` with Random Forests

For Random Forests, the `winner_takes_all` parameter defaults to True
to match the behavior of Breiman's original algorithm. However, setting it to
False can often reduce the model's size and may even improve its quality.

```python
learner = ydf.RandomForestLearner(winner_takes_all=False, label="my_label")
```

### Set a Maximum Model Size

The `maximum_model_size_in_memory_in_bytes` parameter limits the size of the
model in RAM. This is a direct way to control the final model size. Note that
different learning algorithms enforce this limit differently.

```python
# Limit the model to 1 GB in memory
learner = ydf.RandomForestLearner(
    maximum_model_size_in_memory_in_bytes=1e9,
    label="my_label"
)
```

### Increase Shrinkage (GBT only)

The `shrinkage`, or "learning rate", determines how quickly a GBT model learns.
Learning more quickly (`shrinkage` > 0.1) can result in a smaller and faster
model, though it may reduce predictive quality. The default is 0.1; you can try
larger values like 0.15 or even 0.2.
