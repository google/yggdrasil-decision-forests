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

import array
import logging
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.dataset import dataspec as dataspec_lib
from ydf.learner import generic_learner
from ydf.learner import specialized_learners
from ydf.model import export_jax as to_jax
from ydf.model import generic_model
from ydf.model import tree as tree_lib
from ydf.model.gradient_boosted_trees_model import gradient_boosted_trees_model

InternalFeatureItem = to_jax.InternalFeatureItem


def create_dataset(
    columns: List[str], n: int = 1000, seed: Optional[int] = None
) -> Dict[str, Any]:
  """Creates a dataset with random values."""
  if seed is not None:
    np.random.seed(seed)
  data = {
      # Single-dim features
      "f1": np.random.random(size=n),
      "f2": np.random.random(size=n),
      "f3": np.random.random(size=n),
      "f4": np.random.random(size=n),
      "i1": np.random.randint(100, size=n),
      "i2": np.random.randint(100, size=n),
      "c1": np.random.choice(["x", "y", "z"], size=n, p=[0.6, 0.3, 0.1]),
      "c2": np.random.choice([b"x", b"y", b"z"], size=n, p=[0.6, 0.3, 0.1]),
      "c3": np.random.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1]),
      "c4": np.random.randint(2, size=n).astype(np.bool_),
      "b1": np.random.randint(2, size=n).astype(np.bool_),
      "b2": np.random.randint(2, size=n).astype(np.bool_),
      # Cat-set features
      "cs1": [[], ["a", "b", "c"], ["b", "c"], ["a"]] * (n // 4),
      "cs2": [[], [b"a", b"b", b"c"], [b"b", b"c"], [b"a"]] * (n // 4),
      "cs3": [[], [1, 2, 3], [2, 3], [1]] * (n // 4),
      "cs4": [[], [False, True, False], [True, False], [False]] * (n // 4),
      # Multi-dim features
      "multi_f1": np.random.random(size=(n, 5)),
      "multi_f2": np.random.random(size=(n, 5)),
      "multi_i1": np.random.randint(100, size=(n, 5)),
      "multi_c1": np.random.choice(["x", "y", "z"], size=(n, 5)),
      "multi_c2": np.random.choice([b"x", b"y", b"z"], size=(n, 5)),
      "multi_b1": np.random.randint(2, size=(n, 5)).astype(np.bool_),
      # Labels
      "label_class_binary1": np.random.choice([False, True], size=n),
      "label_class_multi1": np.random.choice(["l1", "l2", "l3"], size=n),
      "label_class_multi2": np.random.choice([b"l1", b"l2", b"l3"], size=n),
      "label_regress1": np.random.random(size=n),
  }
  return {k: data[k] for k in columns}


def create_dataset_mnist(
    num_examples: int, begin_example_idx: int = 0
) -> Dict[str, Any]:
  """Creates a binary classification dataset based on MNIST.

  This function cannot be executed in a sandbox test as it downloads a  dataset.

  Args:
    num_examples: Number of examples to extract from the training dataset.
    begin_example_idx: Index of the first example in the training dataset.

  Returns:
    The binary mnist dataset.
  """

  import tensorflow_datasets as tfds  # pylint: disable=import-error,g-import-not-at-top

  tf_ds = tfds.load("mnist", split="train")
  raw_ds = (
      tf_ds.skip(begin_example_idx)
      .batch(num_examples)
      .take(1)
      .as_numpy_iterator()
      .next()
  )
  raw_image_shape = raw_ds["image"].shape
  return {
      "label": raw_ds["label"] >= 5,
      "image": raw_ds["image"].reshape([raw_image_shape[0], -1]),
  }


def create_dataset_ellipse(
    num_examples: int = 1000,
    num_features: int = 3,
    plot_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
  """Creates a binary classification dataset classifying ellipses.

  Args:
    num_examples: Number of generated examples.
    num_features: Number of features.
    plot_path: If set, saves a plot of the examples to this file.

  Returns:
    An ellipse dataset.
  """

  features = np.random.uniform(-1, 1, size=[num_examples, num_features])
  scales = np.array([1.0 + i * 0.4 for i in range(num_features)])
  labels = (
      np.sqrt(np.sum(np.multiply(np.square(features), scales), axis=1)) <= 0.80
  )

  if plot_path:
    colors = ["blue" if l else "red" for l in labels]
    fig, ax = plt.subplots(1, 1)
    ax.scatter(features[:, 0], features[:, 1], color=colors, s=1.5)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(plot_path)

  data = {"label": labels}
  for i in range(num_features):
    data[f"f_{i}"] = features[:, i]
  return data


def plot_ellipse_predictions(
    prediction_fn: Callable[[Dict[str, jax.Array]], jax.Array],
    path: str,
    resolution: int = 200,
    num_features: int = 3,
) -> None:
  """Plots the predictions of a model on the ellipse dataset.

  Args:
    prediction_fn: A function taking a batch of examples, and returning
      predictions.
    path: Save to save the plot.
    resolution: Plotting resolution, for each axis.
    num_features: Number of features.
  """

  # Compute feature values
  vs = np.linspace(-1, 1, resolution)
  tuple_features = np.meshgrid(*((vs,) * num_features))
  raw_features = np.stack([np.reshape(f, [-1]) for f in tuple_features], axis=1)
  dict_features = {
      f"f_{i}": jnp.asarray(raw_features[:, i]) for i in range(num_features)
  }

  # Generate predictions
  predictions = prediction_fn(dict_features)

  disp_predictions = np.reshape(predictions, [resolution] * num_features)
  if len(disp_predictions.shape) > 2:
    disp_predictions = np.mean(
        disp_predictions,
        axis=[i for i in range(2, len(disp_predictions.shape))],
    )

  # Plot predictions
  fig, ax = plt.subplots(1, 1)
  ax.imshow(
      disp_predictions,
      interpolation="none",
      resample=False,
      cmap="coolwarm",
  )
  ax.set_axis_off()
  ax.set_aspect("equal")
  fig.savefig(path)


def check_toy_model(test_self, model):
  test_self.assertEqual(
      model.get_tree(0).pretty(model.data_spec()),
      """\
'f1' >= 2 [score=0 missing=False]
    ├─(pos)─ 'c1' in ['x', 'y'] [score=0 missing=False]
    │        ├─(pos)─ 'c1' in ['x'] [score=0 missing=False]
    │        │        ├─(pos)─ value=1
    │        │        └─(neg)─ value=2
    │        └─(neg)─ value=3
    └─(neg)─ 'f1' >= 1 [score=0 missing=False]
             ├─(pos)─ value=4
             └─(neg)─ value=5
""",
  )

  test_self.assertEqual(
      model.get_tree(1).pretty(model.data_spec()),
      """\
'f2' >= 1.5 [score=0 missing=False]
    ├─(pos)─ value=6
    └─(neg)─ value=7
""",
  )


def create_toy_model(test_self):
  columns = ["f1", "c1", "f2", "c2", "label_regress1"]
  model = specialized_learners.GradientBoostedTreesLearner(
      label="label_regress1",
      task=generic_learner.Task.REGRESSION,
      num_trees=1,
  ).train(create_dataset(columns))
  model.set_initial_predictions([0])
  model.remove_tree(0)

  # pylint: disable=invalid-name
  RegressionValue = tree_lib.RegressionValue
  Leaf = tree_lib.Leaf
  NonLeaf = tree_lib.NonLeaf
  NumericalHigherThanCondition = tree_lib.NumericalHigherThanCondition
  CategoricalIsInCondition = tree_lib.CategoricalIsInCondition
  Tree = tree_lib.Tree
  # pylint: enable=invalid-name

  model.add_tree(
      Tree(
          root=NonLeaf(
              condition=NumericalHigherThanCondition(
                  missing=False, score=0.0, attribute=1, threshold=2.0
              ),
              pos_child=NonLeaf(
                  condition=CategoricalIsInCondition(
                      missing=False,
                      score=0.0,
                      attribute=2,
                      mask=[1, 2],
                  ),
                  pos_child=NonLeaf(
                      condition=CategoricalIsInCondition(
                          missing=False,
                          score=0.0,
                          attribute=2,
                          mask=[1],
                      ),
                      pos_child=Leaf(
                          value=RegressionValue(num_examples=0.0, value=1.0)
                      ),
                      neg_child=Leaf(
                          value=RegressionValue(num_examples=0.0, value=2.0)
                      ),
                  ),
                  neg_child=Leaf(
                      value=RegressionValue(num_examples=0.0, value=3.0)
                  ),
              ),
              neg_child=NonLeaf(
                  condition=NumericalHigherThanCondition(
                      missing=False, score=0.0, attribute=1, threshold=1.0
                  ),
                  pos_child=Leaf(
                      value=RegressionValue(num_examples=0.0, value=4.0)
                  ),
                  neg_child=Leaf(
                      value=RegressionValue(num_examples=0.0, value=5.0)
                  ),
              ),
          )
      )
  )

  model.add_tree(
      Tree(
          root=NonLeaf(
              condition=NumericalHigherThanCondition(
                  missing=False, score=0.0, attribute=3, threshold=1.5
              ),
              pos_child=Leaf(
                  value=RegressionValue(num_examples=0.0, value=6.0)
              ),
              neg_child=Leaf(
                  value=RegressionValue(num_examples=0.0, value=7.0)
              ),
          )
      )
  )
  check_toy_model(test_self, model)
  return model


class JaxModelTest(parameterized.TestCase):

  @parameterized.parameters(
      ((0,), jnp.int8),
      ((0, 1, -1), jnp.int8),
      ((0, 1, 0x7F, -0x80), jnp.int8),
      ((0, 1, 0x7F + 1), jnp.int16),
      ((0, 1, -0x80 - 1), jnp.int16),
      ((0, 1, 0x7FFF), jnp.int16),
      ((0, 1, -0x8000), jnp.int16),
      ((0, 1, 0x7FFF + 1), jnp.int32),
      ((0, 1, -0x8000 - 1), jnp.int32),
      ((0, 1, 0x7FFFFFFF), jnp.int32),
      ((0, 1, -0x80000000), jnp.int32),
  )
  def test_compact_dtype(self, values, expected_dtype):
    self.assertEqual(to_jax.compact_dtype(values), expected_dtype)

    jax_array = to_jax.to_compact_jax_array(values)
    self.assertEqual(jax_array.dtype.type, expected_dtype)
    np.testing.assert_array_equal(jax_array, jnp.array(values, expected_dtype))

  def test_empty_to_compact_jax_array(self):
    jax_array = to_jax.to_compact_jax_array([])
    self.assertEqual(jax_array.dtype.type, jnp.int32)
    np.testing.assert_array_equal(jax_array, jnp.array([0], jnp.int32))

  def test_compact_dtype_non_supported(self):
    with self.assertRaisesRegex(ValueError, "No supported compact dtype"):
      to_jax.compact_dtype((0x80000000,))

  def test_feature_encoder_basic(self):
    feature_encoder = to_jax.FeatureEncoder.build(
        [
            generic_model.InputFeature(
                "f1", dataspec_lib.Semantic.NUMERICAL, 0
            ),
            generic_model.InputFeature(
                "f2", dataspec_lib.Semantic.CATEGORICAL, 1
            ),
            generic_model.InputFeature(
                "f3", dataspec_lib.Semantic.CATEGORICAL, 2
            ),
        ],
        ds_pb.DataSpecification(
            created_num_rows=3,
            columns=(
                ds_pb.Column(
                    name="f1",
                    type=ds_pb.ColumnType.NUMERICAL,
                ),
                ds_pb.Column(
                    name="f2",
                    type=ds_pb.ColumnType.CATEGORICAL,
                    categorical=ds_pb.CategoricalSpec(
                        items={
                            "<OOD>": ds_pb.CategoricalSpec.VocabValue(index=0),
                            "A": ds_pb.CategoricalSpec.VocabValue(index=1),
                            "B": ds_pb.CategoricalSpec.VocabValue(index=2),
                        },
                    ),
                ),
                ds_pb.Column(
                    name="f3",
                    type=ds_pb.ColumnType.CATEGORICAL,
                    categorical=ds_pb.CategoricalSpec(
                        is_already_integerized=True,
                    ),
                ),
                ds_pb.Column(
                    name="f4",
                    type=ds_pb.ColumnType.CATEGORICAL,
                    categorical=ds_pb.CategoricalSpec(
                        items={
                            "<OOD>": ds_pb.CategoricalSpec.VocabValue(index=0),
                            "X": ds_pb.CategoricalSpec.VocabValue(index=1),
                            "Y": ds_pb.CategoricalSpec.VocabValue(index=2),
                        },
                    ),
                ),
            ),
        ),
    )
    self.assertDictEqual(
        feature_encoder.categorical, {"f2": {"<OOD>": 0, "A": 1, "B": 2}}
    )

  def test_feature_encoder_on_model(self):
    columns = [
        "f1",
        "i1",
        "c1",
        "c2",
        "c3",
        "c4",
        "b1",
        "cs1",
        "cs2",
        "cs3",
        "cs4",
        "label_class_binary1",
    ]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary1",
        num_trees=2,
        features=[
            ("c3", dataspec_lib.Semantic.CATEGORICAL),
            ("c4", dataspec_lib.Semantic.CATEGORICAL),
            ("cs1", dataspec_lib.Semantic.CATEGORICAL_SET),
            ("cs2", dataspec_lib.Semantic.CATEGORICAL_SET),
            ("cs3", dataspec_lib.Semantic.CATEGORICAL_SET),
            ("cs4", dataspec_lib.Semantic.CATEGORICAL_SET),
        ],
        include_all_columns=True,
    ).train(create_dataset(columns))
    feature_encoder = to_jax.FeatureEncoder.build(
        model.input_features(), model.data_spec()
    )
    self.assertDictEqual(
        feature_encoder.categorical,
        {
            "c1": {"<OOD>": 0, "x": 1, "y": 2, "z": 3},
            "c2": {"<OOD>": 0, "x": 1, "y": 2, "z": 3},
            "c3": {"<OOD>": 0, "1": 1, "2": 2, "3": 3},
            "c4": {"<OOD>": 0, "false": 1, "true": 2},
            "cs1": {"<OOD>": 0, "a": 1, "b": 2, "c": 3},
            "cs2": {"<OOD>": 0, "a": 1, "b": 2, "c": 3},
            "cs3": {"<OOD>": 0, "1": 1, "2": 2, "3": 3},
            "cs4": {"<OOD>": 0, "false": 1, "true": 2},
        },
    )

    feature_values_py = {
        "f1": [1, 2, 3],
        "c1": ["x", "y", "other"],
        "c2": [b"x", b"y", b"other"],
        "c3": [1, 2, 4],
        "c4": [False, True],
    }
    encoded_features_py = feature_encoder.encode(feature_values_py)
    np.testing.assert_array_equal(
        encoded_features_py["f1"], jnp.asarray([1, 2, 3])
    )
    np.testing.assert_array_equal(
        encoded_features_py["c1"], jnp.asarray([1, 2, 0])
    )
    np.testing.assert_array_equal(
        encoded_features_py["c2"], jnp.asarray([1, 2, 0])
    )
    np.testing.assert_array_equal(
        encoded_features_py["c3"], jnp.asarray([1, 2, 0])
    )
    np.testing.assert_array_equal(
        encoded_features_py["c4"], jnp.asarray([1, 2])
    )

    feature_values_np = {k: np.asarray(v) for k, v in feature_values_py.items()}
    encoded_features_np = feature_encoder.encode(feature_values_np)
    for k in encoded_features_np:
      np.testing.assert_array_equal(
          encoded_features_np[k], encoded_features_py[k]
      )

  def test_feature_encoder_categorical_is_empty(self):
    columns = ["f1", "i1", "label_class_binary1"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary1", num_trees=2
    ).train(create_dataset(columns))
    feature_encoder = to_jax.FeatureEncoder.build(
        model.input_features(), model.data_spec()
    )
    self.assertEmpty(feature_encoder.categorical)

  def test_feature_encoder_unexpected_type(self):
    columns = ["c1", "label_class_binary1"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary1", num_trees=2
    ).train(create_dataset(columns))
    feature_encoder = to_jax.FeatureEncoder.build(
        model.input_features(), model.data_spec()
    )
    encoded_features = feature_encoder.encode({"f1": [1.0, 2.0, 3.0]})
    np.testing.assert_array_equal(
        encoded_features["f1"], jnp.asarray([1.0, 2.0, 3.0])
    )
    with self.assertRaisesRegex(
        ValueError, "Unexpected categorical value type"
    ):
      feature_encoder.encode({"c1": [1.0, 2.0, 3.0]})
    with self.assertRaisesRegex(
        ValueError, "Unexpected categorical value type"
    ):
      feature_encoder.encode({"c1": ["x", "y", None]})
    with self.assertRaisesRegex(
        ValueError, "Unexpected categorical value type"
    ):
      feature_encoder.encode({"c1": [["x", "y"], ["z", "w"]]})


class InternalFeatureSpecTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.basic_mapping = to_jax.InternalFeatureSpec(
        [
            generic_model.InputFeature(
                "n1", dataspec_lib.Semantic.NUMERICAL, 0
            ),
            generic_model.InputFeature(
                "n2", dataspec_lib.Semantic.NUMERICAL, 1
            ),
            generic_model.InputFeature(
                "multidim_n3", dataspec_lib.Semantic.NUMERICAL, 2
            ),
            generic_model.InputFeature(
                "c1", dataspec_lib.Semantic.CATEGORICAL, 4
            ),
            generic_model.InputFeature(
                "c2", dataspec_lib.Semantic.CATEGORICAL, 5
            ),
            generic_model.InputFeature(
                "c3", dataspec_lib.Semantic.CATEGORICAL, 6
            ),
            generic_model.InputFeature(
                "c4", dataspec_lib.Semantic.CATEGORICAL, 7
            ),
            generic_model.InputFeature("b1", dataspec_lib.Semantic.BOOLEAN, 8),
            generic_model.InputFeature("b2", dataspec_lib.Semantic.BOOLEAN, 9),
        ],
        ds_pb.DataSpecification(
            created_num_rows=3,
            columns=(
                ds_pb.Column(
                    name="n1",
                    type=ds_pb.ColumnType.NUMERICAL,
                ),
                ds_pb.Column(
                    name="n2",
                    type=ds_pb.ColumnType.NUMERICAL,
                ),
                ds_pb.Column(
                    name="multidim_n3.0",
                    type=ds_pb.ColumnType.NUMERICAL,
                    is_unstacked=True,
                ),
                ds_pb.Column(
                    name="multidim_n3.1",
                    type=ds_pb.ColumnType.NUMERICAL,
                    is_unstacked=True,
                ),
                ds_pb.Column(
                    name="c1",
                    type=ds_pb.ColumnType.CATEGORICAL,
                ),
                ds_pb.Column(
                    name="c2",
                    type=ds_pb.ColumnType.CATEGORICAL,
                ),
                ds_pb.Column(
                    name="c3",
                    type=ds_pb.ColumnType.CATEGORICAL,
                ),
                ds_pb.Column(
                    name="c4",
                    type=ds_pb.ColumnType.CATEGORICAL,
                ),
                ds_pb.Column(
                    name="b1",
                    type=ds_pb.ColumnType.BOOLEAN,
                ),
                ds_pb.Column(
                    name="b2",
                    type=ds_pb.ColumnType.BOOLEAN,
                ),
            ),
            unstackeds=(
                ds_pb.Unstacked(
                    original_name="multidim_n3",
                    begin_column_idx=2,
                    size=2,
                ),
            ),
        ),
    )

  def test_basic(self):
    self.assertEqual(
        self.basic_mapping.numerical,
        [
            InternalFeatureItem(name="multidim_n3", dim=2),
            InternalFeatureItem(name="n1", dim=1),
            InternalFeatureItem(name="n2", dim=1),
        ],
    )
    self.assertEqual(
        self.basic_mapping.categorical,
        [
            InternalFeatureItem(name="c1", dim=1),
            InternalFeatureItem(name="c2", dim=1),
            InternalFeatureItem(name="c3", dim=1),
            InternalFeatureItem(name="c4", dim=1),
        ],
    )
    self.assertEqual(
        self.basic_mapping.boolean,
        [
            InternalFeatureItem(name="b1", dim=1),
            InternalFeatureItem(name="b2", dim=1),
        ],
    )
    self.assertEqual(self.basic_mapping.inv_numerical, {0: 2, 1: 3, 2: 0, 3: 1})
    self.assertEqual(
        self.basic_mapping.inv_categorical, {4: 0, 5: 1, 6: 2, 7: 3}
    )
    self.assertEqual(self.basic_mapping.inv_boolean, {8: 0, 9: 1})
    self.assertEqual(
        self.basic_mapping.feature_names,
        {"multidim_n3", "n1", "n2", "c1", "c2", "c3", "c4", "b1", "b2"},
    )

  def test_non_supported_type(self):
    with self.assertRaisesRegex(ValueError, "is not supported"):
      to_jax.InternalFeatureSpec(
          [generic_model.InputFeature("n", dataspec_lib.Semantic.HASH, 0)],
          ds_pb.DataSpecification(
              columns=(ds_pb.Column(name="n", type=ds_pb.ColumnType.HASH),)
          ),
      )

  def test_mapping_convert_empty(self):
    with self.assertRaisesRegex(ValueError, "At least one feature"):
      self.basic_mapping.convert_features({})

  def test_mapping_convert_missing(self):
    with self.assertRaisesRegex(
        ValueError, "Expecting dictionary of values with keys"
    ):
      self.basic_mapping.convert_features({"n1": jnp.array([1, 2])})

  def test_mapping_convert_unused(self):
    with self.assertRaisesRegex(
        ValueError, "Expecting dictionary of values with keys"
    ):
      self.basic_mapping.convert_features({
          "n1": jnp.array([1, 2]),
          "n2": jnp.array([3, 4]),
          "multidim_n3": jnp.array([[13, 14], [15, 16]]),
          "c1": jnp.array([5, 6]),
          "c2": jnp.array([7, 8]),
          "c3": jnp.array([9, 10]),
          "c4": jnp.array([11, 12]),
          "b1": jnp.array([True, False]),
          "b2": jnp.array([False, True]),
          "other": jnp.array([1, 2]),
      })

  def test_mapping_convert_wrong_shape_1(self):
    with self.assertRaisesRegex(ValueError, "Expecting dimension"):
      self.basic_mapping.convert_features({
          "n1": jnp.array([1, 2]),
          "n2": jnp.array([3, 4]),
          # multidim_n3 is a 3-dimensional feature, but 1-dimentional values are
          # fed (taking into account the batch size).
          "multidim_n3": jnp.array([13, 14]),
          "c1": jnp.array([5, 6]),
          "c2": jnp.array([7, 8]),
          "c3": jnp.array([9, 10]),
          "c4": jnp.array([11, 12]),
          "b1": jnp.array([True, False]),
          "b2": jnp.array([False, True]),
      })

  def test_mapping_convert_wrong_shape_2(self):
    with self.assertRaisesRegex(ValueError, "Expecting dimension"):
      self.basic_mapping.convert_features({
          "n1": jnp.array([1, 2]),
          "n2": jnp.array([[9, 10], [11, 12]]),
          # multidim_n3 is a 3-dimensional feature, but 2-dimentional values are
          # fed (taking into account the batch size).
          "multidim_n3": jnp.array([[13, 14], [15, 16]]),
          "c1": jnp.array([5, 6]),
          "c2": jnp.array([7, 8]),
          "c3": jnp.array([9, 10]),
          "c4": jnp.array([11, 12]),
          "b1": jnp.array([True, False]),
          "b2": jnp.array([False, True]),
      })

  def test_mapping_convert(self):
    internal_values = self.basic_mapping.convert_features({
        "n1": jnp.array([1, 2]),
        "n2": jnp.array([3, 4]),
        "multidim_n3": jnp.array([[13, 14], [15, 16]]),
        "c1": jnp.array([5, 6]),
        "c2": jnp.array([7, 8]),
        "c3": jnp.array([9, 10]),
        "c4": jnp.array([11, 12]),
        "b1": jnp.array([True, False]),
        "b2": jnp.array([False, True]),
    })
    np.testing.assert_array_equal(
        internal_values.numerical,
        jnp.array([[13.0, 14.0, 1.0, 3.0], [15.0, 16.0, 2.0, 4.0]]),
    )
    np.testing.assert_array_equal(
        internal_values.categorical, jnp.array([[5, 7, 9, 11], [6, 8, 10, 12]])
    )
    np.testing.assert_array_equal(
        internal_values.boolean, jnp.array([[True, False], [False, True]])
    )


class NodexIdxTest(parameterized.TestCase):

  @parameterized.parameters(
      (2, None, 2, 3, 0),
      (3, None, 2, 3, 1),
      (None, 3, 2, 3, -1),
      (None, 4, 2, 3, -2),
  )
  def test_node_offset(
      self,
      non_leaf_node: Optional[int],
      leaf_node: Optional[int],
      begin_non_leaf_node: int,
      begin_leaf_node: int,
      expected: int,
  ):
    self.assertEqual(
        to_jax.NodeIdx(non_leaf_node=non_leaf_node, leaf_node=leaf_node).offset(
            to_jax.BeginNodeIdx(
                non_leaf_node=begin_non_leaf_node, leaf_node=begin_leaf_node
            )
        ),
        expected,
    )

  @parameterized.parameters(
      ((), 0, []),
      ((), 4, [False, False, False, False]),
      ((1, 3), 4, [False, True, False, True]),
      ((0, 1, 2, 3), 4, [True, True, True, True]),
  )
  def test_categorical_list_to_bitmap(
      self, items: Sequence[int], size: int, expected: List[bool]
  ):
    self.assertEqual(
        to_jax._categorical_list_to_bitmap(
            ds_pb.Column(
                categorical=ds_pb.CategoricalSpec(number_of_unique_values=size)
            ),
            items,
        ),
        expected,
    )


class InternalForestTest(parameterized.TestCase):

  def test_categorical_list_to_bitmap_invalid(self):
    with self.assertRaisesRegex(ValueError, "Invalid item"):
      to_jax._categorical_list_to_bitmap(
          ds_pb.Column(
              categorical=ds_pb.CategoricalSpec(number_of_unique_values=2)
          ),
          [2],
      )

  def test_internal_forest_on_manual(self):
    model = create_toy_model(self)
    internal_forest = to_jax.InternalForest(model)

    self.assertEqual(
        internal_forest.feature_spec.numerical,
        [
            InternalFeatureItem(name="f1", dim=1),
            InternalFeatureItem(name="f2", dim=1),
        ],
    )
    self.assertEqual(
        internal_forest.feature_spec.categorical,
        [
            InternalFeatureItem(name="c1", dim=1),
            InternalFeatureItem(name="c2", dim=1),
        ],
    )
    self.assertEqual(internal_forest.feature_spec.boolean, [])
    self.assertEqual(internal_forest.feature_spec.inv_numerical, {1: 0, 3: 1})
    self.assertEqual(internal_forest.feature_spec.inv_categorical, {2: 0, 4: 1})
    self.assertEqual(internal_forest.feature_spec.inv_boolean, {})
    self.assertEqual(
        internal_forest.feature_spec.feature_names, {"f1", "f2", "c1", "c2"}
    )

    self.assertEqual(internal_forest.num_trees(), 2)
    self.assertNotEmpty(internal_forest.feature_encoder.categorical)

    self.assertEqual(
        internal_forest.leaf_outputs,
        array.array("f", [5.0, 4.0, 3.0, 2.0, 1.0, 7.0, 6.0]),
    )
    self.assertEqual(
        internal_forest.split_features, array.array("l", [0, 0, 0, 0, 1])
    )

    def bitcast_uint32_to_float(x):
      return float(
          jax.lax.bitcast_convert_type(
              jnp.array(x, dtype=jnp.int32), jnp.float32
          )
      )

    self.assertEqual(
        internal_forest.split_parameters,
        array.array(
            "f",
            [
                2.0,
                1.0,
                bitcast_uint32_to_float(0),
                bitcast_uint32_to_float(4),
                1.5,
            ],
        ),
    )
    self.assertEqual(
        internal_forest.negative_children, array.array("l", [1, -1, -3, -4, -1])
    )
    self.assertEqual(
        internal_forest.positive_children, array.array("l", [2, -2, 3, -5, -2])
    )
    self.assertEqual(
        internal_forest.condition_types,
        array.array(
            "l",
            [
                to_jax.ConditionType.GREATER_THAN,
                to_jax.ConditionType.GREATER_THAN,
                to_jax.ConditionType.IS_IN,
                to_jax.ConditionType.IS_IN,
                to_jax.ConditionType.GREATER_THAN,
            ],
        ),
    )
    self.assertEqual(
        internal_forest.begin_non_leaf_nodes, array.array("l", [0, 4])
    )
    self.assertEqual(internal_forest.begin_leaf_nodes, array.array("l", [0, 5]))
    self.assertEqual(
        internal_forest.catgorical_mask,
        array.array("b", [False, True, True, False, False, True, False, False]),
    )
    self.assertEqual(internal_forest.max_depth, 3)

  def test_internal_forest_on_model(self):
    columns = ["f1", "i1", "c1", "label_regress1"]
    model = specialized_learners.RandomForestLearner(
        label="label_regress1",
        task=generic_learner.Task.REGRESSION,
        num_trees=10,
        max_depth=5,
    ).train(create_dataset(columns))

    internal_forest = to_jax.InternalForest(model)
    self.assertEqual(internal_forest.num_trees(), 10)
    self.assertNotEmpty(internal_forest.feature_encoder.categorical)


@absltest.skip("Broken by JAX v0.4.35")
class ToJaxTest(parameterized.TestCase):

  @parameterized.parameters(
      ([], {}, []),
      ([1, 2, 3], {1: 0, 2: 1, 3: 2}, [0, 1, 2]),
      ([1, 3], {1: 0, 3: 1}, [0, 1]),
  )
  def test_densify_conditions(
      self, src_conditions, expected_mapping, expected_dense_conditions
  ):
    mapping, dense_conditions = to_jax._densify_conditions(src_conditions)
    self.assertEqual(mapping, expected_mapping)
    self.assertEqual(
        dense_conditions, array.array("l", expected_dense_conditions)
    )

  @parameterized.named_parameters(
      (
          "gbt_regression_num",
          ["f1", "f2"],
          "label_regress1",
          generic_learner.Task.REGRESSION,
          False,
          specialized_learners.GradientBoostedTreesLearner,
      ),
      (
          "gbt_regression_num_cat",
          ["f1", "f2", "c1", "c2"],
          "label_regress1",
          generic_learner.Task.REGRESSION,
          True,
          specialized_learners.GradientBoostedTreesLearner,
      ),
      (
          "gbt_class_binary_num_cat",
          ["f1", "f2", "c1", "c2", "label_class_binary1"],
          "label_class_binary1",
          generic_learner.Task.CLASSIFICATION,
          True,
          specialized_learners.GradientBoostedTreesLearner,
      ),
      (
          "gbt_class_binary_num_cat_tfl",
          ["f1", "f2", "c1", "c2", "label_class_binary1"],
          "label_class_binary1",
          generic_learner.Task.CLASSIFICATION,
          True,
          specialized_learners.GradientBoostedTreesLearner,
          None,
          True,
          True,
      ),
      (
          "gbt_class_multi_num_cat1",
          ["f1", "f2", "c1", "c2", "label_class_multi1"],
          "label_class_multi1",
          generic_learner.Task.CLASSIFICATION,
          True,
          specialized_learners.GradientBoostedTreesLearner,
      ),
      (
          "gbt_class_multi_num_cat2",
          ["f1", "f2", "c1", "c2", "label_class_multi2"],
          "label_class_multi2",
          generic_learner.Task.CLASSIFICATION,
          True,
          specialized_learners.GradientBoostedTreesLearner,
      ),
      (
          "gbt_regression_num_multidim",
          ["f1", "multi_f1"],
          "label_regress1",
          generic_learner.Task.REGRESSION,
          False,
          specialized_learners.GradientBoostedTreesLearner,
      ),
      (
          "gbt_regression_num_oblique",
          ["f1", "f2", "f3", "f4"],
          "label_regress1",
          generic_learner.Task.REGRESSION,
          False,
          specialized_learners.GradientBoostedTreesLearner,
          {
              "split_axis": "SPARSE_OBLIQUE",
              "sparse_oblique_normalization": "STANDARD_DEVIATION",
          },
          # TODO: b/364685350 - Re-enable when compiler fix oss sync
          # is released.
          False
      ),
  )
  def test_to_jax_function(
      self,
      features: List[str],
      label: str,
      task: generic_learner.Task,
      has_encoding: bool,
      learner,
      learner_kwargs=None,
      test_tf_conversion: bool = True,
      test_tf_lite: bool = False,
  ):

    if learner_kwargs is None:
      learner_kwargs = {}
    else:
      learner_kwargs = learner_kwargs.copy()

    if learner == specialized_learners.GradientBoostedTreesLearner:
      learner_kwargs["validation_ratio"] = 0.0

    # Create YDF model
    columns = features + [label]
    model = learner(
        label=label,
        task=task,
        **learner_kwargs,
    ).train(create_dataset(columns, 1000, seed=1))
    self.assertIsInstance(
        model, gradient_boosted_trees_model.GradientBoostedTreesModel
    )

    # Golden predictions
    test_ds = create_dataset(columns, 10, seed=2)
    ydf_predictions = model.predict(test_ds)

    # Convert model to a jax function
    with jax.numpy_dtype_promotion("strict"):
      kwargs = {}
      if test_tf_lite:
        kwargs["compatibility"] = "TFL"
      jax_model = to_jax.to_jax_function(model, **kwargs)
      assert bool(jax_model.encoder.categorical) == has_encoding

      # Generate Jax predictions
      del test_ds[label]
      input_values = jax_model.encoder(test_ds)
      jax_predictions = jax_model.predict(input_values)

    # Test predictions
    # Note: This test sometime fails with OOS build.
    np.testing.assert_allclose(
        jax_predictions,
        ydf_predictions,
        rtol=1e-5,
        atol=1e-5,
    )

    if not test_tf_conversion:
      return

    # Convert to a TensorFlow function
    tf_model = tf.Module()
    tf_model.my_call = tf.function(
        jax2tf.convert(jax_model.predict, with_gradient=False),
        jit_compile=True,
        autograph=False,
    )

    # Check TF predictions
    tf_predictions = tf_model.my_call(input_values)
    np.testing.assert_allclose(
        tf_predictions,
        ydf_predictions,
        rtol=1e-5,
        atol=1e-5,
    )

    # Saved and restore the TensorFlow function
    with tempfile.TemporaryDirectory() as tempdir:
      tf.saved_model.save(tf_model, tempdir)
      restored_tf_model = tf.saved_model.load(tempdir)

    # Check TF predictions from the restored model
    restored_tf_predictions = restored_tf_model.my_call(input_values)
    np.testing.assert_allclose(
        restored_tf_predictions,
        jax_predictions,
        rtol=1e-5,
        atol=1e-5,
    )

    if test_tf_lite:
      # TODO: Re-enable TFLite conversion when TFLite tfl.gather_nd
      # issue is resolved.
      return

      # Test TFLite conversion

      # Generate the input specs.
      # Note: TFL uses the spec names to generate the argument names.
      # Note: "from_tensor" does not implement the automatic JAX->TF conversion.
      tf_input_specs = {
          k: tf.TensorSpec.from_tensor(tf.constant(v), name=k)
          for k, v in input_values.items()
      }

      concrete_predict = tf_model.my_call.get_concrete_function(tf_input_specs)
      converter = tf.lite.TFLiteConverter.from_concrete_functions(
          [concrete_predict], tf_model
      )
      # TODO: Remove when TFL support all the YDF->JAX required ops.
      converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
          tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
      ]
      tflite_model = converter.convert()
      interpreter = tf.lite.Interpreter(model_content=tflite_model)
      interpreter.allocate_tensors()
      tfl_predict = interpreter.get_signature_runner("serving_default")

      logging.info("input_values:\n%s", input_values)
      logging.info("tfl_predict:\n%s", tfl_predict)
      logging.info("concrete_predict:\n%s", concrete_predict)

      tfl_predictions = tfl_predict(**input_values)["output_0"]
      np.testing.assert_allclose(
          ydf_predictions,
          tfl_predictions,
          rtol=1e-5,
          atol=1e-5,
      )

  def test_dataset_ellipse(self):
    with tempfile.TemporaryDirectory() as tempdir:
      _ = create_dataset_ellipse(
          1000, plot_path=os.path.join(tempdir, "ds.png")
      )

      def prediction_fn(features: Dict[str, jax.Array]) -> jax.Array:
        return features["f_0"] >= 0.1

      plot_ellipse_predictions(
          prediction_fn, os.path.join(tempdir, "predictions.png")
      )

  @parameterized.named_parameters(
      ("leaves_ellipse_axis_aligned", "ellipse", True, False),
      # ("leaves_mnist", "mnist", True , False),   # Skip in sandboxed test
      ("leaves_ellipse_oblique", "ellipse", True, True),
  )
  def test_fine_tune_model(
      self, dataset: str, leaves_as_params: bool, oblique: bool
  ):

    # Note: Optax cannot be imported in python 3.8.
    import optax  # pylint: disable=import-error,g-import-not-at-top

    # Make datasets
    label = "label"

    if dataset == "ellipse":
      train_ds = create_dataset_ellipse(1000)
      test_ds = create_dataset_ellipse(10000)
      finetune_ds = create_dataset_ellipse(10000)
    elif dataset == "mnist":
      train_ds = create_dataset_mnist(1000)
      test_ds = create_dataset_mnist(10000, 1000)
      finetune_ds = create_dataset_mnist(10000, 1000 + 10000)
    else:
      assert False

    kwargs = {}
    if oblique:
      kwargs["split_axis"] = "SPARSE_OBLIQUE"
      kwargs["sparse_oblique_normalization"] = "STANDARD_DEVIATION"

    # Train a model with YDF
    model = specialized_learners.GradientBoostedTreesLearner(
        label=label, **kwargs
    ).train(train_ds)

    # Evaluate the YDF model
    pre_tuned_ydf_accuracy = model.evaluate(test_ds).accuracy
    logging.info("pre_tuned_ydf_accuracy: %s", pre_tuned_ydf_accuracy)

    # Convert the model to JAX and evaluate it
    jax_model = to_jax.to_jax_function(
        model,
        apply_activation=False,
        leaves_as_params=leaves_as_params,
    )

    # Check parameter values
    if leaves_as_params:
      self.assertContainsSubset(
          ["leaf_values", "initial_predictions"], jax_model.params
      )

    def to_jax_array(d):
      """Converts a numpy array into a jax array."""
      return {k: jnp.asarray(v) for k, v in d.items()}

    jax_test_ds = to_jax_array(test_ds)
    jax_finetune_ds = to_jax_array(finetune_ds)

    @jax.jit
    def compute_predictions(state, batch):
      batch = batch.copy()
      batch.pop(label)
      return jax_model.predict(batch, state)

    @jax.jit
    def compute_accuracy(state, batch):
      batch = batch.copy()
      labels = batch.pop(label)
      predictions = jax_model.predict(batch, state)
      return jnp.mean((predictions >= 0.0) == labels)

    @jax.jit
    def compute_loss(state, batch):
      batch = batch.copy()
      labels = batch.pop(label)
      logits = jax_model.predict(batch, state)
      loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
      return loss

    pre_tuned_jax_test_accuracy = float(
        compute_accuracy(jax_model.params, jax_test_ds)
    )
    logging.info("pre_tuned_jax_test_accuracy: %s", pre_tuned_jax_test_accuracy)
    self.assertAlmostEqual(
        pre_tuned_jax_test_accuracy, pre_tuned_ydf_accuracy, delta=1.0e-5
    )

    # Finetune the JAX model
    assert jax_model.params is not None
    self.assertSetEqual(
        set(jax_model.params), set(["leaf_values", "initial_predictions"])
    )

    # Evaluate the fine-tuned JAX model
    optimizer = optax.adam(0.001)

    @jax.jit
    def train_step(opt_state, mdl_state, batch):
      loss, grads = jax.value_and_grad(compute_loss)(mdl_state, batch)
      updates, opt_state = optimizer.update(grads, opt_state)
      mdl_state = optax.apply_updates(mdl_state, updates)
      return opt_state, mdl_state, loss

    tf_finetune_ds = tf.data.Dataset.from_tensor_slices(finetune_ds).batch(100)

    mdl_state = jax_model.params
    opt_state = optimizer.init(mdl_state)
    for epoch_idx in range(10):

      test_loss = compute_loss(mdl_state, jax_test_ds)
      finetune_loss = compute_loss(mdl_state, jax_finetune_ds)
      test_accuracy = compute_accuracy(mdl_state, jax_test_ds)
      finetune_accuracy = compute_accuracy(mdl_state, jax_finetune_ds)

      logging.info("epoch_idx: %s", epoch_idx)
      logging.info("\ttest_loss: %s", test_loss)
      logging.info("\tfinetune_loss: %s", finetune_loss)
      logging.info("\ttest_accuracy: %s", test_accuracy)
      logging.info("\tfinetune_accuracy: %s", finetune_accuracy)

      for batch in tf_finetune_ds:
        opt_state, mdl_state, loss = train_step(
            opt_state, mdl_state, to_jax_array(batch)
        )
        del loss

    post_tuned_jax_test_accuracy = float(
        compute_accuracy(mdl_state, jax_test_ds)
    )
    logging.info(
        "post_tuned_jax_test_accuracy: %s", post_tuned_jax_test_accuracy
    )
    # The model quality improve by at least 0.008 of accuracy.
    self.assertGreaterEqual(
        post_tuned_jax_test_accuracy, pre_tuned_jax_test_accuracy + 0.008
    )

    # Update the YDF model with the finetuned parameters
    model.update_with_jax_params(mdl_state)

    # Check the weights have been updated
    new_jax_model = to_jax.to_jax_function(
        model, leaves_as_params=leaves_as_params
    )
    np.testing.assert_allclose(
        mdl_state["initial_predictions"],
        new_jax_model.params["initial_predictions"],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        mdl_state["leaf_values"],
        new_jax_model.params["leaf_values"],
        rtol=1e-5,
        atol=1e-5,
    )

    # Check the predictions of the updated YDF model
    finetuned_ydf_predictions = model.predict(
        test_ds,
    )
    finetuned_jax_predictions = compute_predictions(mdl_state, jax_test_ds)
    np.testing.assert_allclose(
        jax.nn.sigmoid(finetuned_jax_predictions),
        finetuned_ydf_predictions,
        rtol=1e-5,
        atol=1e-5,
    )

  def test_update_with_jax_params_manual(self):
    model = create_toy_model(self)
    check_toy_model(self, model)

    jax_model = to_jax.to_jax_function(model, leaves_as_params=True)
    to_jax.update_with_jax_params(model, jax_model.params)

    # Nothing have changed yet
    check_toy_model(self, model)

    np.testing.assert_allclose(
        jax_model.params["leaf_values"],
        [5.0, 4.0, 3.0, 2.0, 1.0, 7.0, 6.0],
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        jax_model.params["initial_predictions"],
        [0.0],
        rtol=1e-5,
        atol=1e-5,
    )

    jax_model.params["leaf_values"] = jnp.asarray(
        [1.0, 2.0, 3.0, 4.0, 5, 6.0, 7.0], jnp.float32
    )
    jax_model.params["initial_predictions"] = jnp.asarray([1.0], jnp.float32)
    to_jax.update_with_jax_params(model, jax_model.params)

    np.testing.assert_allclose(
        model.initial_predictions(),
        jax_model.params["initial_predictions"],
        rtol=1e-5,
        atol=1e-5,
    )

    new_jax_model = to_jax.to_jax_function(model, leaves_as_params=True)
    np.testing.assert_allclose(
        jax_model.params["initial_predictions"],
        new_jax_model.params["initial_predictions"],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        jax_model.params["leaf_values"],
        new_jax_model.params["leaf_values"],
        rtol=1e-5,
        atol=1e-5,
    )

    self.assertEqual(
        model.get_tree(0).pretty(model.data_spec()),
        """\
'f1' >= 2 [score=0 missing=False]
    ├─(pos)─ 'c1' in ['x', 'y'] [score=0 missing=False]
    │        ├─(pos)─ 'c1' in ['x'] [score=0 missing=False]
    │        │        ├─(pos)─ value=5
    │        │        └─(neg)─ value=4
    │        └─(neg)─ value=3
    └─(neg)─ 'f1' >= 1 [score=0 missing=False]
             ├─(pos)─ value=2
             └─(neg)─ value=1
""",
    )

    self.assertEqual(
        model.get_tree(1).pretty(model.data_spec()),
        """\
'f2' >= 1.5 [score=0 missing=False]
    ├─(pos)─ value=7
    └─(neg)─ value=6
""",
    )


class BitmapTest(parameterized.TestCase):

  @parameterized.parameters(
      ([], []),
      ([1], [0b0001]),
      ([0], [0b0000]),
      ([0, 1], [0b0010]),
      ([1, 0, 0, 1] * 8, [0b_1001_1001_1001_1001_1001_1001_1001_1001]),
      (
          [1, 0, 0, 1] * 8 + [1],
          [0b_1001_1001_1001_1001_1001_1001_1001_1001, 0b0001],
      ),
  )
  def test_compress_bitmap(self, src, expected):
    np.testing.assert_equal(to_jax.compress_bitmap(src), expected)

  @parameterized.parameters(
      ([1], 0, True),
      ([1, 0, 0, 1], 0, True),
      ([1, 0, 0, 1], 1, False),
      ([1, 0, 0, 1], 2, False),
      ([1, 0, 0, 1], 3, True),
      ([0] * 31 + [1], 31, True),
      ([0] * 31 + [0], 31, False),
      ([0] * 32 + [1], 32, True),
      ([0] * 32 + [0], 32, False),
  )
  def test_get_bit(self, bitmap, query, expected):
    np.testing.assert_equal(
        to_jax.get_bit(jnp.array(to_jax.compress_bitmap(bitmap)), query),
        jnp.bool_(expected),
    )


if __name__ == "__main__":
  if sys.version_info < (3, 9):
    print("JAX is not supported anymore on python <= 3.8. Skipping JAX tests.")
  else:
    absltest.main()
