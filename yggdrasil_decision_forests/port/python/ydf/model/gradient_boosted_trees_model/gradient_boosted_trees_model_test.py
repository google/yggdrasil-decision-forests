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

"""Tests for the Gradient Boosted trees models."""

import os
from typing import Dict, Tuple

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
import pandas as pd

from ydf.dataset import dataspec
from ydf.learner import custom_loss
from ydf.learner import specialized_learners
from ydf.model import generic_model
from ydf.model import model_lib
from ydf.model.gradient_boosted_trees_model import gradient_boosted_trees_model
from ydf.model.tree import condition as condition_lib
from ydf.model.tree import node as node_lib
from ydf.model.tree import tree as tree_lib
from ydf.model.tree import value as value_lib
from ydf.utils import test_utils

RegressionValue = value_lib.RegressionValue
Leaf = node_lib.Leaf
NonLeaf = node_lib.NonLeaf
NumericalHigherThanCondition = condition_lib.NumericalHigherThanCondition
Tree = tree_lib.Tree


def load_model(
    name: str,
    directory: str = "model",
) -> gradient_boosted_trees_model.GradientBoostedTreesModel:
  path = os.path.join(test_utils.ydf_test_data_path(), directory, name)
  return model_lib.load_model(path)


class GradientBoostedTreesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # This model is a classification model for pure serving.
    self.adult_binary_class_gbdt = load_model("adult_binary_class_gbdt")

    # This model is a classification model with full training logs.
    self.gbt_adult_base_with_na = load_model(
        "gbt_adult_base_with_na", directory="golden"
    )

    self.iris_multi_class_gbdt = load_model("iris_multi_class_gbdt")
    self.synthetic_ranking_gbdt = load_model("synthetic_ranking_gbdt")
    self.abalone_regression_gbdt = load_model("abalone_regression_gbdt")

  def test_input_feature_names(self):
    self.assertEqual(
        self.adult_binary_class_gbdt.input_feature_names(),
        [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
        ],
    )

  def test_input_features(self):
    InputFeature = generic_model.InputFeature
    NUMERICAL = dataspec.Semantic.NUMERICAL
    CATEGORICAL = dataspec.Semantic.CATEGORICAL
    self.assertEqual(
        self.adult_binary_class_gbdt.input_features(),
        [
            InputFeature("age", NUMERICAL, 0),
            InputFeature("workclass", CATEGORICAL, 1),
            InputFeature("fnlwgt", NUMERICAL, 2),
            InputFeature("education", CATEGORICAL, 3),
            InputFeature("education_num", CATEGORICAL, 4),
            InputFeature("marital_status", CATEGORICAL, 5),
            InputFeature("occupation", CATEGORICAL, 6),
            InputFeature("relationship", CATEGORICAL, 7),
            InputFeature("race", CATEGORICAL, 8),
            InputFeature("sex", CATEGORICAL, 9),
            InputFeature("capital_gain", NUMERICAL, 10),
            InputFeature("capital_loss", NUMERICAL, 11),
            InputFeature("hours_per_week", NUMERICAL, 12),
            InputFeature("native_country", CATEGORICAL, 13),
        ],
    )

  def test_task(self):
    self.assertEqual(
        self.adult_binary_class_gbdt.task(), generic_model.Task.CLASSIFICATION
    )

  def test_label_classes(self):
    self.assertEqual(
        self.adult_binary_class_gbdt.label_classes(), ["<=50K", ">50K"]
    )

  def test_label(self):
    self.assertEqual(self.adult_binary_class_gbdt.label(), "income")

  def test_validation_loss(self):
    validation_loss = self.adult_binary_class_gbdt.validation_loss()
    self.assertAlmostEqual(validation_loss, 0.573842942, places=6)

  def test_validation_loss_if_no_validation_dataset(self):
    dataset = {"x": np.array([0, 0, 1, 1]), "y": np.array([0, 0, 0, 1])}
    model = specialized_learners.GradientBoostedTreesLearner(
        label="y", validation_ratio=0.0, num_trees=2
    ).train(dataset)
    validation_loss = model.validation_loss()
    self.assertIsNone(validation_loss)

  def test_initial_predictions(self):
    initial_predictions = self.adult_binary_class_gbdt.initial_predictions()
    np.testing.assert_allclose(initial_predictions, [-1.1630996])

  @parameterized.parameters(
      "adult_binary_class_gbdt",
      "iris_multi_class_gbdt",
      "abalone_regression_gbdt",
  )
  def test_set_initial_predictions(self, model_name):
    model = load_model(model_name)
    initial_predictions = model.initial_predictions()
    model.set_initial_predictions(initial_predictions * 2.0)
    np.testing.assert_allclose(
        initial_predictions * 2, model.initial_predictions()
    )

  def test_validation_evaluation_empty(self):
    dataset = {
        "x1": np.array([0, 0, 0, 1, 1, 1]),
        "y": np.array([0, 0, 0, 0, 1, 1]),
    }
    model = specialized_learners.GradientBoostedTreesLearner(
        label="y",
        num_trees=1,
        max_depth=4,
        min_examples=1,
        validation_ratio=0.0,
    ).train(dataset)
    self.assertIsInstance(
        model, gradient_boosted_trees_model.GradientBoostedTreesModel
    )
    validation_evaluation = model.validation_evaluation()
    self.assertIsNone(validation_evaluation)

  def test_validation_evaluation_no_training_logs(self):
    validation_evaluation = self.adult_binary_class_gbdt.validation_evaluation()
    self.assertIsNotNone(validation_evaluation)
    self.assertIsNone(validation_evaluation.accuracy)
    self.assertAlmostEqual(validation_evaluation.loss, 0.57384294)

  def test_validation_evaluation_with_content(self):
    validation_evaluation = self.gbt_adult_base_with_na.validation_evaluation()
    self.assertIsNotNone(validation_evaluation)
    self.assertAlmostEqual(validation_evaluation.accuracy, 0.8498403)

  def test_variable_importances_stored_in_model(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "synthetic_ranking_gbdt_numerical",
    )
    model = model_lib.load_model(model_path)
    variable_importances = model.variable_importances()
    self.assertEqual(
        variable_importances,
        {
            "NUM_NODES": [
                (355.0, "num_2"),
                (326.0, "num_0"),
                (248.0, "num_1"),
                (193.0, "num_3"),
            ],
            "INV_MEAN_MIN_DEPTH": [
                (0.54955206094026765, "num_0"),
                (0.43300866801748344, "num_2"),
                (0.21987296105251422, "num_1"),
                (0.20886402442940008, "num_3"),
            ],
            "SUM_SCORE": [
                (331.52462868355724, "num_0"),
                (297.70595154801595, "num_2"),
                (103.86176226850876, "num_1"),
                (52.43193327602421, "num_3"),
            ],
            "NUM_AS_ROOT": [
                (35.0, "num_0"),
                (12.0, "num_2"),
                (1.0, "num_3"),
            ],
        },
    )

  def test_variable_importances_not_stored_in_model(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "synthetic_ranking_gbdt",
    )
    model = model_lib.load_model(model_path)
    variable_importances = model.variable_importances()
    self.assertEqual(
        variable_importances,
        {
            "INV_MEAN_MIN_DEPTH": [
                (0.823529411764706, "cat_str_0"),
                (0.3409269442262372, "num_0"),
                (0.33853354134165364, "num_2"),
                (0.2407099278979479, "cat_str_1"),
                (0.16596558317399618, "num_1"),
                (0.15816326530612249, "cat_int_0"),
                (0.15645277577505406, "num_3"),
                (0.1550553769203287, "cat_int_1"),
            ],
            "NUM_AS_ROOT": [
                (11.0, "cat_str_0"),
                (2.0, "num_0"),
                (1.0, "num_2"),
            ],
            "NUM_NODES": [
                (128.0, "cat_str_1"),
                (101.0, "cat_str_0"),
                (65.0, "num_0"),
                (64.0, "num_2"),
                (26.0, "num_1"),
                (15.0, "cat_int_0"),
                (11.0, "cat_int_1"),
                (10.0, "num_3"),
            ],
            "SUM_SCORE": [
                (357.5797439698363, "cat_str_0"),
                (290.9118405326153, "num_0"),
                (225.41135752020637, "cat_str_1"),
                (183.18018738733372, "num_2"),
                (37.17157095257426, "num_1"),
                (19.275285203708336, "cat_int_0"),
                (10.371429289167281, "cat_int_1"),
                (8.130908778170124, "num_3"),
            ],
        },
    )

  def test_activation(self):
    self.assertEqual(
        self.adult_binary_class_gbdt.activation(),
        custom_loss.Activation.SIGMOID,
    )
    self.assertEqual(
        self.iris_multi_class_gbdt.activation(),
        custom_loss.Activation.SOFTMAX,
    )
    self.assertEqual(
        self.synthetic_ranking_gbdt.activation(),
        custom_loss.Activation.IDENTITY,
    )
    self.assertEqual(
        self.abalone_regression_gbdt.activation(),
        custom_loss.Activation.IDENTITY,
    )

  def test_num_trees_per_iterations(self):
    self.assertEqual(self.adult_binary_class_gbdt.num_trees_per_iteration(), 1)

  def test_predict_distance(self):
    dataset = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        ),
        nrows=500,
    )

    distances = self.adult_binary_class_gbdt.distance(dataset)
    logging.info("distances:\n%s", distances)
    self.assertEqual(distances.shape, (dataset.shape[0], dataset.shape[0]))

    # Find in "dataset2", the example most similar to "dataset1[0]".
    most_similar_example_idx = np.argmin(distances[0, :])
    logging.info("most_similar_example_idx: %s", most_similar_example_idx)
    logging.info("Seed example:\n%s", dataset.iloc[0])
    logging.info(
        "Most similar example:\n%s", dataset.iloc[most_similar_example_idx]
    )

    # High likelihood that the labels are the same (true in this example).
    self.assertEqual(
        dataset.iloc[most_similar_example_idx]["income"],
        dataset.iloc[0]["income"],
    )

  def test_model_inspector_get_valid_tree(self):
    self.assertEqual(self.adult_binary_class_gbdt.num_trees(), 68)
    self.assertLen(
        self.adult_binary_class_gbdt.get_all_trees(),
        self.adult_binary_class_gbdt.num_trees(),
    )

    tree = self.adult_binary_class_gbdt.get_tree(1)
    self.assertFalse(tree.root.is_leaf)
    # Validated with: external/ydf_cc/yggdrasil_decision_forests/cli:show_model
    self.assertIsNotNone(tree.root)
    assert isinstance(tree.root, node_lib.NonLeaf)
    self.assertEqual(
        tree.root.condition,
        condition_lib.CategoricalIsInCondition(
            missing=False,
            score=3275.003662109375,
            attribute=5,
            mask=[2, 3, 4, 5, 6, 7],
        ),
    )
    self.assertEqual(
        tree.root.value,
        value_lib.RegressionValue(
            value=-0.0006140652694739401, num_examples=0.0
        ),
    )

  def test_model_inspector_get_wrong_tree(self):
    with self.assertRaisesRegex(ValueError, "Invalid tree index"):
      _ = self.adult_binary_class_gbdt.get_tree(-1)
    with self.assertRaisesRegex(ValueError, "Invalid tree index"):
      _ = self.adult_binary_class_gbdt.get_tree(
          self.adult_binary_class_gbdt.num_trees()
      )

  def test_model_inspector_print_tree(self):
    tree = self.adult_binary_class_gbdt.get_tree(1)
    test_utils.golden_check_string(
        self,
        tree.pretty(self.adult_binary_class_gbdt.data_spec()),
        os.path.join(test_utils.pydf_test_data_path(), "adult_gbt_tree_0.txt"),
    )


class EditModelTest(absltest.TestCase):

  def create_model_and_dataset(
      self,
  ) -> Tuple[
      gradient_boosted_trees_model.GradientBoostedTreesModel,
      Dict[str, np.ndarray],
  ]:
    dataset = {
        "x1": np.array([0, 0, 0, 1, 1, 1]),
        "x2": np.array([1, 1, 0, 0, 1, 1]),
        "y": np.array([0, 0, 0, 0, 1, 1]),
    }
    model = specialized_learners.GradientBoostedTreesLearner(
        label="y",
        num_trees=1,
        max_depth=4,
        apply_link_function=False,
        min_examples=1,
    ).train(dataset)
    assert isinstance(
        model, gradient_boosted_trees_model.GradientBoostedTreesModel
    )
    return model, dataset

  def test_create_model_and_dataset(self):
    model, dataset = self.create_model_and_dataset()
    tree = model.get_tree(0)
    self.assertEqual(model.num_trees(), 1)
    self.assertEqual(
        tree.pretty(model.data_spec()),
        """\
'x1' >= 0.5 [score=0.11111 missing=True]
    ├─(pos)─ 'x2' >= 0.5 [score=0.22222 missing=True]
    │        ├─(pos)─ value=0.3
    │        └─(neg)─ value=-0.15 sd=4.069e-05
    └─(neg)─ value=-0.15 sd=4.069e-05
""",
    )
    bias = -0.693147
    npt.assert_almost_equal(model.initial_predictions(), [bias], decimal=4)
    npt.assert_almost_equal(
        model.predict(dataset),
        [
            bias - 0.15,
            bias - 0.15,
            bias - 0.15,
            bias - 0.15,
            bias + 0.3,
            bias + 0.3,
        ],
        decimal=4,
    )

  def test_set_tree(self):
    model, dataset = self.create_model_and_dataset()
    tree = model.get_tree(0)
    assert isinstance(tree.root, node_lib.NonLeaf)
    assert isinstance(tree.root.pos_child.pos_child, node_lib.Leaf)
    assert isinstance(
        tree.root.pos_child.pos_child.value, value_lib.RegressionValue
    )
    tree.root.pos_child.pos_child.value.value = 0.1

    expected_tree_repr = """\
'x1' >= 0.5 [score=0.11111 missing=True]
    ├─(pos)─ 'x2' >= 0.5 [score=0.22222 missing=True]
    │        ├─(pos)─ value=0.1
    │        └─(neg)─ value=-0.15 sd=4.069e-05
    └─(neg)─ value=-0.15 sd=4.069e-05
"""

    self.assertEqual(tree.pretty(model.data_spec()), expected_tree_repr)
    model.set_tree(0, tree)
    self.assertEqual(model.num_trees(), 1)
    self.assertEqual(
        model.get_tree(0).pretty(model.data_spec()), expected_tree_repr
    )
    bias = -0.693147
    npt.assert_almost_equal(
        model.predict(dataset),
        [
            bias - 0.15,
            bias - 0.15,
            bias - 0.15,
            bias - 0.15,
            bias + 0.1,
            bias + 0.1,
        ],
        decimal=4,
    )

  def test_add_tree(self):
    model, dataset = self.create_model_and_dataset()

    tree = Tree(
        root=NonLeaf(
            condition=NumericalHigherThanCondition(
                missing=False, score=3.0, attribute=1, threshold=0.6
            ),
            pos_child=Leaf(value=RegressionValue(num_examples=1.0, value=2.0)),
            neg_child=Leaf(value=RegressionValue(num_examples=1.0, value=-2.0)),
        )
    )
    expected_tree_repr = """\
'x1' >= 0.6 [score=3 missing=False]
    ├─(pos)─ value=2
    └─(neg)─ value=-2
"""

    self.assertEqual(tree.pretty(model.data_spec()), expected_tree_repr)
    model.add_tree(tree)
    self.assertEqual(model.num_trees(), 2)
    self.assertEqual(
        model.get_tree(1).pretty(model.data_spec()), expected_tree_repr
    )
    bias = -0.693147
    npt.assert_almost_equal(
        model.predict(dataset),
        [
            bias - 0.15 - 2.0,
            bias - 0.15 - 2.0,
            bias - 0.15 - 2.0,
            bias - 0.15 + 2.0,
            bias + 0.3 + 2.0,
            bias + 0.3 + 2.0,
        ],
        decimal=4,
    )

  def test_remove_tree(self):
    model, dataset = self.create_model_and_dataset()
    model.remove_tree(0)
    self.assertEqual(model.num_trees(), 0)
    bias = -0.693147
    npt.assert_almost_equal(
        model.predict(dataset),
        [bias] * 6,
        decimal=4,
    )

  def test_invalid_inference(self):
    dataset = {
        "x1": np.array([0, 0, 2, 2, 1, 1]),
        "y": np.array([0, 0, 1, 1, 2, 2]),
    }
    model = specialized_learners.GradientBoostedTreesLearner(
        label="y",
        num_trees=1,
        max_depth=4,
        apply_link_function=False,
        min_examples=1,
    ).train(dataset)
    assert isinstance(
        model, gradient_boosted_trees_model.GradientBoostedTreesModel
    )
    self.assertEqual(model.num_trees(), 3)
    self.assertEqual(
        model.get_tree(0).pretty(model.data_spec()),
        """\
'x1' >= 0.5 [score=0.22222 missing=True]
    ├─(pos)─ value=-0.15 sd=4.069e-05
    └─(neg)─ value=0.3
""",
    )
    self.assertEqual(
        model.get_tree(1).pretty(model.data_spec()),
        """\
'x1' >= 1.5 [score=0.22222 missing=False]
    ├─(pos)─ value=0.3
    └─(neg)─ value=-0.15 sd=4.069e-05
""",
    )
    self.assertEqual(
        model.get_tree(2).pretty(model.data_spec()),
        """\
'x1' >= 0.5 [score=0.055556 missing=True]
    ├─(pos)─ 'x1' >= 1.5 [score=0.25 missing=False]
    │        ├─(pos)─ value=-0.15 sd=4.069e-05
    │        └─(neg)─ value=0.3
    └─(neg)─ value=-0.15 sd=4.069e-05
""",
    )

    tree = Tree(
        root=NonLeaf(
            condition=NumericalHigherThanCondition(
                missing=False, score=3.0, attribute=1, threshold=0.6
            ),
            pos_child=Leaf(value=RegressionValue(num_examples=1.0, value=2.0)),
            neg_child=Leaf(value=RegressionValue(num_examples=1.0, value=-2.0)),
        )
    )
    expected_tree_repr = """\
'x1' >= 0.6 [score=3 missing=False]
    ├─(pos)─ value=2
    └─(neg)─ value=-2
"""
    self.assertEqual(tree.pretty(model.data_spec()), expected_tree_repr)
    model.add_tree(tree)
    self.assertEqual(model.num_trees(), 4)

    with self.assertRaisesRegex(
        ValueError, "Invalid number of trees in the gradient boosted tree"
    ):
      _ = model.predict(dataset)

  def test_add_tree_to_empty_forest(self):
    dataset = {
        "x1": np.array([0, 1]),
        "x2": np.array([0, 1]),
        "x3": np.array([0, 1]),
        "y": np.array([0, 1]),
    }
    model = specialized_learners.GradientBoostedTreesLearner(
        label="y", num_trees=0
    ).train(dataset)
    assert isinstance(
        model, gradient_boosted_trees_model.GradientBoostedTreesModel
    )
    self.assertEqual(model.num_trees(), 0)
    self.assertSequenceEqual(model.input_feature_names(), ["x1", "x2", "x3"])

    tree = Tree(
        root=NonLeaf(
            condition=NumericalHigherThanCondition(
                missing=False, score=3.0, attribute=2, threshold=0.6
            ),
            pos_child=Leaf(value=RegressionValue(num_examples=1.0, value=2.0)),
            neg_child=Leaf(value=RegressionValue(num_examples=1.0, value=-2.0)),
        )
    )
    model.add_tree(tree)
    self.assertEqual(model.num_trees(), 1)
    self.assertSequenceEqual(model.input_feature_names(), ["x1", "x2", "x3"])

    model.remove_tree(0)
    self.assertEqual(model.num_trees(), 0)
    self.assertSequenceEqual(model.input_feature_names(), ["x1", "x2", "x3"])


if __name__ == "__main__":
  absltest.main()
