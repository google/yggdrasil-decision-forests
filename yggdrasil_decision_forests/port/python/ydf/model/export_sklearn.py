# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Import and export Scikit-Learn models from/to YDF."""

import dataclasses
import enum
import functools
from typing import Any, Dict, List, Optional, TypeVar, Union, Sequence
import numpy as np

from ydf.learner import generic_learner
from ydf.learner import specialized_learners
from ydf.model import generic_model
from ydf.model import tree as tree_lib
from ydf.model.gradient_boosted_trees_model import gradient_boosted_trees_model
from ydf.model.isolation_forest_model import isolation_forest_model
from ydf.model.random_forest_model import random_forest_model

# pytype: disable=import-error
# pylint: disable=g-import-not-at-top
try:
  from sklearn import base
  from sklearn import dummy
  from sklearn import ensemble
  from sklearn import tree
except ImportError as exc:
  raise ImportError("Cannot import sklearn") from exc
# pylint: enable=g-import-not-at-top
# pytype: enable=import-error


# The column idx=0 is reserved for the label in YDF models.
_LABEL_COLUMN_OFFSET = 1


@dataclasses.dataclass(frozen=True)
class InternalOptions:
  """Internal options for the conversion.

  Attributes:
    label_name: Column name of the created label.
    feature_name: Column name of the created feature.
  """

  label_name: str
  feature_name: str


class TaskType(enum.Enum):
  """The type of task that a scikit-learn model performs."""

  UNKNOWN = 1
  SCALAR_REGRESSION = 2
  SINGLE_LABEL_CLASSIFICATION = 3
  ANOMALY_DETECTION = 4


ScikitLearnModel = TypeVar("ScikitLearnModel", bound=base.BaseEstimator)
ScikitLearnTree = TypeVar("ScikitLearnTree", bound=tree.BaseDecisionTree)


def from_sklearn(
    sklearn_model: ScikitLearnModel,
    label_name: str = "label",
    feature_name: str = "features",
) -> generic_model.GenericModel:
  """Converts a tree-based scikit-learn model to a YDF model."""
  if not hasattr(sklearn_model, "n_features_in_"):
    raise ValueError(
        "Scikit-Learn model must be fit to data before converting."
    )
  return _sklearn_to_ydf_model(
      sklearn_model,
      InternalOptions(label_name=label_name, feature_name=feature_name),
  )


def _gen_fake_features(num_features: int, num_examples: int = 2):
  return np.zeros(shape=[num_examples, num_features])


@functools.singledispatch
def _sklearn_to_ydf_model(
    sklearn_model: ScikitLearnModel, options: InternalOptions
) -> generic_model.GenericModel:
  """Builds a YDF model from the given scikit-learn model."""
  del options
  raise NotImplementedError(
      f"Can't build a YDF model for {type(sklearn_model)}"
  )


@_sklearn_to_ydf_model.register(tree.DecisionTreeRegressor)
@_sklearn_to_ydf_model.register(tree.ExtraTreeRegressor)
def _(
    sklearn_model: ScikitLearnTree, options: InternalOptions
) -> generic_model.GenericModel:
  """Converts a single scikit-learn regression tree to a YDF model."""
  ydf_model = specialized_learners.RandomForestLearner(
      label=options.label_name,
      task=generic_learner.Task.REGRESSION,
      num_trees=0,
  ).train(
      {
          options.label_name: [0.0, 1.0],
          options.feature_name: _gen_fake_features(
              sklearn_model.n_features_in_
          ),
      },
      verbose=0,
  )
  assert isinstance(ydf_model, random_forest_model.RandomForestModel)
  ydf_tree = convert_sklearn_tree_to_ydf_tree(
      sklearn_model, task=generic_learner.Task.REGRESSION
  )
  ydf_model.add_tree(ydf_tree)
  return ydf_model


@_sklearn_to_ydf_model.register(tree.DecisionTreeClassifier)
@_sklearn_to_ydf_model.register(tree.ExtraTreeClassifier)
def _(
    sklearn_model: ScikitLearnTree, options: InternalOptions
) -> generic_model.GenericModel:
  """Converts a single scikit-learn classification tree to a YDF model."""
  ydf_model = specialized_learners.RandomForestLearner(
      label=options.label_name,
      task=generic_learner.Task.CLASSIFICATION,
      num_trees=0,
  ).train(
      {
          options.label_name: [str(c) for c in sklearn_model.classes_],
          options.feature_name: _gen_fake_features(
              sklearn_model.n_features_in_, len(sklearn_model.classes_)
          ),
      },
      verbose=0,
  )
  assert isinstance(ydf_model, random_forest_model.RandomForestModel)
  ydf_tree = convert_sklearn_tree_to_ydf_tree(
      sklearn_model, task=generic_learner.Task.CLASSIFICATION
  )
  ydf_model.add_tree(ydf_tree)
  return ydf_model


@_sklearn_to_ydf_model.register(ensemble.ExtraTreesRegressor)
@_sklearn_to_ydf_model.register(ensemble.RandomForestRegressor)
def _(
    sklearn_model: Union[
        ensemble.ExtraTreesRegressor,
        ensemble.RandomForestRegressor,
    ],
    options: InternalOptions,
) -> generic_model.GenericModel:
  """Converts a forest regression model into a YDF model."""

  ydf_model = specialized_learners.RandomForestLearner(
      label=options.label_name,
      task=generic_learner.Task.REGRESSION,
      num_trees=0,
  ).train(
      {
          options.label_name: [0.0, 1.0],
          options.feature_name: _gen_fake_features(
              sklearn_model.n_features_in_
          ),
      },
      verbose=0,
  )
  assert isinstance(ydf_model, random_forest_model.RandomForestModel)
  for sklearn_tree in sklearn_model.estimators_:
    ydf_tree = convert_sklearn_tree_to_ydf_tree(
        sklearn_tree, task=generic_learner.Task.REGRESSION
    )
    ydf_model.add_tree(ydf_tree)
  return ydf_model


@_sklearn_to_ydf_model.register(ensemble.ExtraTreesClassifier)
@_sklearn_to_ydf_model.register(ensemble.RandomForestClassifier)
def _(
    sklearn_model: Union[
        ensemble.ExtraTreesClassifier,
        ensemble.RandomForestClassifier,
    ],
    options: InternalOptions,
) -> generic_model.GenericModel:
  """Converts a forest classification model into a YDF model."""

  ydf_model = specialized_learners.RandomForestLearner(
      label=options.label_name,
      task=generic_learner.Task.CLASSIFICATION,
      num_trees=0,
  ).train(
      {
          options.label_name: [str(c) for c in sklearn_model.classes_],
          options.feature_name: _gen_fake_features(
              sklearn_model.n_features_in_, len(sklearn_model.classes_)
          ),
      },
      verbose=0,
  )
  assert isinstance(
      ydf_model,
      random_forest_model.RandomForestModel,
  )
  for sklearn_tree in sklearn_model.estimators_:
    ydf_tree = convert_sklearn_tree_to_ydf_tree(
        sklearn_tree, task=generic_learner.Task.CLASSIFICATION
    )
    ydf_model.add_tree(ydf_tree)
  return ydf_model


@_sklearn_to_ydf_model.register(ensemble.IsolationForest)
def _(
    sklearn_model: ensemble.IsolationForest, options: InternalOptions
) -> generic_model.GenericModel:
  """Converts a single scikit-learn iso-forest to a YDF model."""
  ydf_model = specialized_learners.IsolationForestLearner(
      task=generic_learner.Task.ANOMALY_DETECTION, num_trees=0
  ).train(
      {
          options.feature_name: _gen_fake_features(
              sklearn_model.n_features_in_
          ),
      },
      verbose=0,
  )
  sklearn_examples_per_tree: int = int(sklearn_model._max_samples)  # pylint: disable=protected-access
  ydf_model._model.set_num_examples_per_tree(sklearn_examples_per_tree)  # pylint: disable=protected-access
  assert isinstance(ydf_model, isolation_forest_model.IsolationForestModel)

  for sklearn_tree, attribute_mapping in zip(
      sklearn_model.estimators_, sklearn_model.estimators_features_
  ):
    ydf_tree = convert_sklearn_tree_to_ydf_tree(
        sklearn_tree,
        attribute_mapping=attribute_mapping.tolist(),
        task=generic_learner.Task.ANOMALY_DETECTION,
    )
    ydf_model.add_tree(ydf_tree)
  return ydf_model


@_sklearn_to_ydf_model.register(ensemble.GradientBoostingRegressor)
def _(
    sklearn_model: ensemble.GradientBoostingRegressor, options: InternalOptions
) -> generic_model.GenericModel:
  """Converts a gradient boosting regression model into a YDF model."""

  if isinstance(sklearn_model.init_, dummy.DummyRegressor):
    # If the initial estimator is a DummyRegressor, then it predicts a constant
    # which can be passed to GradientBoostedTreeBuilder as a bias.
    init_pytree = None
    bias = sklearn_model.init_.constant_[0][0]
  elif isinstance(sklearn_model.init_, tree.DecisionTreeRegressor):
    # If the initial estimator is a DecisionTreeRegressor, we add it as the
    # first tree in the ensemble and set the bias to zero. We could also support
    # other tree-based initial estimators (e.g. RandomForest), but this seems
    # like a niche enough use case that we don't for the moment.
    init_pytree = convert_sklearn_tree_to_ydf_tree(
        sklearn_model.init_, task=generic_learner.Task.REGRESSION
    )
    bias = 0.0
  elif sklearn_model.init_ == "zero":
    init_pytree = None
    bias = 0.0
  else:
    raise ValueError(
        "The initial estimator must be either a DummyRegressor"
        "or a DecisionTreeRegressor, but got"
        f"{type(sklearn_model.init_)}."
    )

  ydf_model = specialized_learners.GradientBoostedTreesLearner(
      label=options.label_name,
      task=generic_learner.Task.REGRESSION,
      num_trees=0,
  ).train(
      {
          options.label_name: [0.0, 1.0],
          options.feature_name: _gen_fake_features(
              sklearn_model.n_features_in_
          ),
      },
      verbose=0,
  )
  assert isinstance(
      ydf_model, gradient_boosted_trees_model.GradientBoostedTreesModel
  )

  ydf_model.set_initial_predictions([bias])

  if init_pytree:
    ydf_model.add_tree(init_pytree)

  for weak_learner in sklearn_model.estimators_.ravel():
    ydf_tree = convert_sklearn_tree_to_ydf_tree(
        weak_learner,
        weight=sklearn_model.learning_rate,
        task=generic_learner.Task.REGRESSION,
    )
    ydf_model.add_tree(ydf_tree)
  return ydf_model


def convert_sklearn_tree_to_ydf_tree(
    sklearn_tree: ScikitLearnTree,
    task: generic_learner.Task,
    weight: Optional[float] = None,
    attribute_mapping: Optional[Sequence[int]] = None,
) -> tree_lib.Tree:
  """Converts a scikit-learn decision tree into a YDF tree.

  Args:
    sklearn_tree: a scikit-learn decision tree.
    task: The task of the model.
    weight: an optional weight to apply to the values of the leaves in the tree.
      This is intended for use when converting gradient boosted tree models.
    attribute_mapping: Index of the attributes used as input features for this
      sktree.

  Returns:
    a YDF tree that has the same structure as the scikit-learn tree.
  """
  try:
    sklearn_tree_data = sklearn_tree.tree_.__getstate__()
  except AttributeError as e:
    raise ValueError(
        "Scikit-Learn model must be fit to data before converting."
    ) from e

  if hasattr(sklearn_tree, "n_classes_") and sklearn_tree.n_outputs_ == 1:
    pass  # A classification model
  elif sklearn_tree.n_outputs_ == 1:
    pass  # A regression model
  else:
    raise ValueError(
        "This model type if not supported. `ydf.from_sklearn` only support"
        " scalar regression, single-label classification and isolation forests."
    )

  field_names = sklearn_tree_data["nodes"].dtype.names

  nodes = []
  # For each node
  for node_properties, node_output in zip(
      sklearn_tree_data["nodes"],
      sklearn_tree_data["values"],
  ):
    # Dictionary of node properties (e.g. "left_child", "threshold") except for
    # the node output value.
    node = {
        field_name: field_value
        for field_name, field_value in zip(field_names, node_properties)
    }

    common_kwargs = {"num_examples": node["weighted_n_node_samples"]}

    # Add the node output value to the dictionary of properties.
    if task == generic_learner.Task.REGRESSION:
      scaling_factor = weight if weight else 1.0
      node["value"] = tree_lib.RegressionValue(
          value=node_output[0][0] * scaling_factor, **common_kwargs
      )
    elif task == generic_learner.Task.CLASSIFICATION:
      # Normalise to probabilities if we have a classification tree.
      assert weight is None
      probabilities = list(node_output[0] / node_output[0].sum())
      node["value"] = tree_lib.ProbabilityValue(
          probability=probabilities, **common_kwargs
      )

    elif task == generic_learner.Task.ANOMALY_DETECTION:
      assert weight is None
      node["value"] = tree_lib.AnomalyDetectionValue(
          num_examples_without_weight=node["n_node_samples"], **common_kwargs
      )
    else:
      raise ValueError(
          "Only scalar regression and single-label classification are "
          "supported."
      )
    nodes.append(node)

  root_node = _convert_sklearn_node_to_ydf_node(
      # The root node has index zero.
      node_index=0,
      nodes=nodes,
      attribute_mapping=attribute_mapping,
      task=task,
  )
  return tree_lib.Tree(root_node)


def _convert_sklearn_node_to_ydf_node(
    node_index: int,
    task: generic_learner.Task,
    nodes: List[Dict[str, Any]],
    attribute_mapping: Optional[Sequence[int]],
) -> tree_lib.AbstractNode:
  """Converts a node within a scikit-learn tree into a YDF node."""
  if node_index == -1:
    raise ValueError("Bad node idx")

  node = nodes[node_index]
  is_leaf = node["left_child"] == -1

  if is_leaf:
    return tree_lib.Leaf(value=node["value"])

  neg_child = _convert_sklearn_node_to_ydf_node(
      node_index=node["left_child"],
      task=task,
      nodes=nodes,
      attribute_mapping=attribute_mapping,
  )
  pos_child = _convert_sklearn_node_to_ydf_node(
      node_index=node["right_child"],
      task=task,
      nodes=nodes,
      attribute_mapping=attribute_mapping,
  )

  attribute = node["feature"]
  if attribute_mapping:
    attribute = attribute_mapping[attribute]
  else:
    attribute += _LABEL_COLUMN_OFFSET

  return tree_lib.NonLeaf(
      condition=tree_lib.NumericalHigherThanCondition(
          attribute=attribute,
          threshold=node["threshold"],
          missing=False,
          score=0.0,
      ),
      pos_child=pos_child,
      neg_child=neg_child,
  )
