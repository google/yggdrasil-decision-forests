# How to improve a model?

This page lists methods to improve the quality, speed, and size of YDF models.

**Note:** Understanding how decision forests work helps understanding and using
those algorithms. Learn more about decision forests algorithms in our
[Decision Forests class](https://developers.google.com/machine-learning/decision-forests)
on the Google Developer website.

## Random Forest or Gradient Boosted Trees?

**Random Forests** (RF) and **Gradient Boosted Trees** (GBT) are two different
algorithms for training decision forests. Each algorithm has its own set of
strengths and weaknesses. At a high level, RFs are less prone to overfitting
than GBTs, making them a good choice for small datasets and datasets with a
large number of input features. On the other hand, GBTs learn more efficiently
than RFs. On large datasets, GBTs can lead to significantly stronger models.
Additionally, GBT models are often much smaller and allow for faster inference
than comparable RF models. **Warning:** Both algorithms have hyperparameters in
common, such as the number of trees and the maximum tree depth. However, these
hyperparameters play a different role in each algorithm and should be tuned
accordingly. For example, the maximum tree depth of a GBT is typically between 3
and 8, while it is rarely less than 16 in an RF.

## Automated hyper-parameter tuning

Automated hyperparameter tuning is a simple but expensive solution to improve
the quality of a model. When full hyper-parameter tuning is too expensive,
combining hyper-parameter tuning and manual tuning (explained in the next
sections) is a good solution.

## Hyper-parameter templates

The default hyperparameters of YDF learners are set to reproduce the originally
published algorithms. As such, they are not necessarily optimized for
performance. This can lead to reasonable, but not optimal, results. For the user
to benefit from the latest YDF algorithm without having understood those
hyper-parameters and without having to run hyper-parameter tuning, YDF have
pre-configured hyper-parameter templates.

## Number of trees (GBT and RF)

The size and representational power of a model are controlled by the number of
trees. In the case of Random Forest, the model quality improves with the
addition of more trees until a plateau. Increasing the number of trees in Random
Forest does not lead to overfitting. For the Random Forest algorithm to function
properly, there should be a large number of trees (200 is the minimum, 1000 is a
good number) with a high depth (16 is a good default value). For Gradient
Boosted Trees, the model quality improves with the addition of more trees until
the model begins to overfit. At this point, the model training will stop
automatically. Increasing the number of trees while also increasing model
regularization (such as shrinkage or attribute sampling) typically improves the
model quality.

## Best first global growth strategy (GBT only)

By default, trees are trained using a greedy divide-and-conquer approach. An
alternative and more modern method is to train trees globally (called
growing_strategy=`BEST_FIRST_GLOBAL`). While it does not always perform better,
it is generally worth trying. When using global tree optimization, the maximum
number of nodes should be tuned instead of the maximum depth.

## Oblique splits (GBT and RF)

By default, trees are "orthogonal" i.e. each split/condition tests a single
feature. By opposition, conditions in oblique trees can use multiple features.
Oblique splits generally improve performances.

Oblique trees are more expensive to train. The `num_projections_exponent`
parameter plays an important role in the training time and final model quality
(1 is cheap, 2 is better but more expensive). See the
[training configuration](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto#L152)
for more details.

**Training config:**

```proto
decision_tree {
    sparse_oblique_split {
       num_projections_exponent : 1.5
       normalization: NONE
    }
  }
```

**Generic hyper-parameter:**

```python
split_axis = "SPARSE_OBLIQUE"
sparse_oblique_num_projections_exponent = 2
sparse_oblique_normalization = "MIN_MAX"
```

## Random Categorical splits (GBT and RF)

By default, categorical splits are learned with the CART algorithm. Training
categorical split with the Random algorithm can improve the model performances
at the expense of model size.

**Training config:**

```proto
decision_tree {
    categorical {
      random {}
    }
  }
```

**Generic hyper-parameter:**

```python
categorical_algorithm = "RANDOM"
```

## Hessian splits (GBT only)

By default, splits are scored with a first-order approximation of the gradient.
Using a second-order approximation, also called hessian splots, can improve the
performance.

**Training config:**

```proto
use_hessian_gain: true
```

**Generic hyper-parameter:**

```python
use_hessian_gain = "true"
```

## Disabling the validation dataset (GBT only)

By default, GBT extracts a sample of the training dataset to build a validation
dataset (default to 10%). For small datasets, it might be good to use all the
data for training (and therefore disable early-stopping). In this case, the
`num_trees` parameter should be tuned.

**Training config:**

```proto
validation_set_ratio: 0.0
early_stopping: NONE
```

**Generic hyper-parameter:**

```python
validation_ratio = 0.0
early_stopping = "NONE"
```

## Disabling winner take all (RF only)

By default, each tree in an RF is voting for a single class. When disabling
winner takes all, each tree is voting for the distribution of classes. This
generally improves the model.

**Training config:**

```proto
winner_take_all_inference: false
```

**Generic hyper-parameter:**

```python
winner_take_all = "false"
```

## Super Learners

Following are examples of GBT and RF training configurations with all the method
listed above:

```proto
learner: "GRADIENT_BOOSTED_TREES"

[yggdrasil_decision_forests.model.gradient_boosted_trees.proto.gradient_boosted_trees_config] {
  num_trees: 1000
  use_hessian_gain: true
  validation_set_ratio: 0.0
  early_stopping: NONE
  decision_tree {
    growing_strategy_best_first_global { max_num_nodes: 64 }
    sparse_oblique_split {}
    categorical { random {} }
  }
}
```

```proto
learner: "RANDOM_FOREST"

[yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
  num_trees: 1000
  winner_take_all_inference: false
  decision_tree {
    sparse_oblique_split {}
    categorical { random {} }
  }
}
```

## Improving the size of the model

The size of a model is critical in some applications. YDF models range from a
few KB to a few GB. The following sections list some way you can reduce the size
of a model.

### 1. Switch from a Random Forest to a Gradient Boosted Trees

Random Forests models are significantly larger and slower than Gradient Boosted
Trees.

**TensorFlow Decision Forests code**

```python
# learner = ydf.RandomForestLearner()
learner = ydf.GradientBoostedTreesLearner()
```

### 2. Remove model meta-data

YDF models contain meta-data used for model interpretation and debugging. This
meta-data is not used for model inference and can be discarded to decrease the
model size.

The meta-data can be removed with the argument `pure_serving_model=True`.

**TensorFlow Decision Forests code**

```python
learner = ydf.RandomForestLearner(pure_serving_model=True)
```

**Yggdrasil Decision Forests training configuration**

```python
pure_serving_model: true
```

The meta-data of an already existing model can be removed with the the
`edit_model` CLI tool:

```shell
# Remove the meta-data from the model
./edit_model --input=/tmp/model_with_metadata --output=/tmp/model_without_metadata --pure_serving=true

# Look at the size of the model
du -h /tmp/model_with_metadata
du -h /tmp/model_without_metadata
```

Results:

```
528K /tmp/model_with_metadata
264K /tmp/model_without_metadata
```

### 3. Ensure that the model is correctly trained

Unique ID-like features (e.g., user id) that cannot be generated about will make
the model grows without benefit. Make sure to not include such type of input
features.

ID-like features can be spotted using the variable importance. They generally
have high "number of nodes" variable importance while all the other variable
importance measures are low.

### 4. Reduce the number of trees

The `num_trees` parameter controls the number of trees in the model. Reducing
this parameter will decrease the size of the model at the expense of the model
quality.

### 5. Disable `winner_take_all_inference` with Random Forests

The `winner_take_all_inference` parameters (true by default) can make Random
Forest models are large. Try disabling it.

**TensorFlow Decision Forests code**

```python
model = tfdf.keras.RandomForestModel(winner_take_all=False)
```

**Yggdrasil Decision Forests training configuration**

```python
winner_take_all_inference: false
```

### 6. Set `maximum_model_size_in_memory_in_bytes`

The `maximum_model_size_in_memory_in_bytes` parameter controls the maximize size
of the model when loaded in memory. By setting this value, you can control the
final size of the model.

**TensorFlow Decision Forests code**

```python
model = tfdf.keras.RandomForestModel(maximum_model_size_in_memory_in_bytes=10e+9  # 10GB)
```

**Yggdrasil Decision Forests training configuration**

```python
maximum_model_size_in_memory_in_bytes: 10e+9  # 10GB
```
