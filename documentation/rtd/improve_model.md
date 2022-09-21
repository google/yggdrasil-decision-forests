# How to improve models

This page list methods to improve the quality and speed of YDF models.

``` {note}
Learn more about decision forests algorithms in our
[Decision Forests class](https://developers.google.com/machine-learning/decision-forests)
on the Google Developer website. Understanding how decision forests work helps
understanding and using those algorithms.
```

## Random Forest or Gradient Boosted Trees?

**Random Forests** (RF) and **Gradient Boosted Trees** (GBT) are two very
different algorithms to train decision forests. Each algorithm comes with its
own set of strengths and weaknesses.

At a high level, **Random Forests suffer less from overfitting** than Gradient
Boosted Trees, making Random Forests a great choice for small datasets and
datasets with a large number of input features.

On the other hand, **Gradient Boosted Trees learn more efficiently** than Random
Forests. On large datasets, Gradient Boosted Trees lead to significantly
stronger models. Furthermore, GBT models are often much smaller and allow for
faster inference than comparable RF models.

If the model speed or size is critical, GBT should be selected. In other cases,
it is worth trying out both RF and GBT and selecting the best model.

``` {warning}
Both algorithms have hyperparameters in common. For example, the number of trees
and the maximum tree depth. While those hyper-parameters are common, they play a
different role and should be tuned differently depending on the algorithm. For
example, the *maximum tree depth* of a GBT is generally in between 3 and 8,
while it is rarely less than 16 in an RF.
```

## Automated hyper-parameter tuning

Automated hyperparameter tuning is a simple but expensive solution to improve
the quality of a model. See the
[C++/CLI hyper-parameter tuning](cli_user_manual.md#automated-hyper-parameter-tuning)
or
[TensorFlow Decision Forests hyper-parameter tuning](https://www.tensorflow.org/decision_forests/tutorials/automatic_tuning_colab)
pages for more details.

When full hyper-parameter tuning is too expensive, combining hyper-parameter
tuning and manual tuning (explained in the next sections) is a good solutions.

## Hyper-parameter templates

The default hyper-parameters of YDF learners are set to reproduce the
corresponding originally published algorithm. In addition, YDF offers backward
compatibility of hyper-parameters: Running a learner configured with a set of
given hyper-parameters always returns the same model (modulo changes to the
pseudo-random number generators). For those two reasons, the default
hyperparameters are giving reasonable but not optimal results.

For the user to benefit from the latest YDF algorithm without having understood
those hyper-parameters and without having to run hyper-parameter tuning, YDF
introduces a hyper-parameter template system. Those hyper-parameter templates
contain improved hyper-parameters. Those templates are available in the
[hyper_parameters](hyper_parameters) page.

~~~ {note}
In TensorFlow Decision Forests, the hyperparameter template can be specified with the `hyperparameter_template` model constructor argument. The next example train a Gradient Boosted Trees model with the `benchmark_rank1` template.

```python
# A good template of hyper-parameters.
model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1")
model.fit(train_ds)
```

~~~

### Number of trees (GBT and RF)

The number of trees controls the size and power of expression of a model.

In the case of Random Forest, increasing the number of trees increase the model
quality until a plateau. Increasing the number of trees in Random Forests does
not cause overfitting. For the Random Forest algorithm to work correctly, there
should be a lot of trees (200 is a minimum, 1000 is a good number) with a high
depth (16 is a good default value).

In the case of Gradient Boosted Trees, increasing the number of trees increase
the model quality until the model starts overfitting. At this point, the model
training stops automatically. Increasing the number of trees while increasing
model regularization (e.g., shrinkage or attribute sampling) generally improves
the quality of the model.

**Training config:**

```proto
num_trees: 2000
```

**Generic hyper-parameter:**

```python
num_trees = 2000
```

## Best first global growth strategy (GBT only)

By default, trees are built using a greedy divide-and-conquer algorithm. Growing
the tree globally can improve the model performance. In this case, the **maximum
number of nodes** can be tuned and the **maximum tree depth** can be set to
infinity.

**Training config:**

```proto
decision_tree {
    growing_strategy_best_first_global {
      max_num_nodes: 64
      max_depth: 128
    }
  }
```

**Generic hyper-parameter:**

```python
growing_strategy = "BEST_FIRST_GLOBAL"
max_num_nodes = 64
max_depth=128
```

## Oblique splits (GBT and RF)

By default, trees are "orthogonal" i.e. each split/condition tests a single
feature. By opposition, conditions in oblique trees can use multiple features.
Oblique splits generally improve performances.

Oblique trees are more expensive to train. The num_projections_exponent
parameter plays an important role in the training time and final model quality
(1 is cheap, 2 is good but expensive). See the
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
sparse_oblique_normalization = 64
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

By default, splits are trained with a first-order approximation of the gradient.
Second-order approximation can improve the performance.

**Training config:**

```proto
use_hessian_gain: true
```

**Generic hyper-parameter:**

```python
use_hessian_gain = "true"
```

### Disabling the validation dataset (GBT only)

By default, GBT extracts a sample of the training dataset to build a validation
dataset (default to 10%). For small datasets, it might be good to use all the
data for training (and therefore disable early-stopping). In this case, the
`num_trees` parameter should be tuned. This operation can both improve or hurt
the model.

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

### Disabling winner take all (RF only)

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

### Super Learners

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
