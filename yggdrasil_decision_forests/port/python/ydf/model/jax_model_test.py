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
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.dataset import dataspec as dataspec_lib
from ydf.learner import generic_learner
from ydf.learner import specialized_learners
from ydf.model import export_jax as to_jax
from ydf.model import generic_model
from ydf.model import tree as tree_lib


def create_dataset(columns: List[str], n: int = 1000) -> Dict[str, Any]:
  """Creates a dataset with random values."""
  np.random.seed(10)
  data = {
      # Single-dim features
      "f1": np.random.random(size=n),
      "f2": np.random.random(size=n),
      "i1": np.random.randint(100, size=n),
      "i2": np.random.randint(100, size=n),
      "c1": np.random.choice(["x", "y", "z"], size=n, p=[0.6, 0.3, 0.1]),
      "b1": np.random.randint(2, size=n).astype(np.bool_),
      "b2": np.random.randint(2, size=n).astype(np.bool_),
      # Cat-set features
      "cs1": [[], ["a", "b", "c"], ["b", "c"], ["a"]] * (n // 4),
      # Multi-dim features
      "multi_f1": np.random.random(size=(n, 5)),
      "multi_f2": np.random.random(size=(n, 5)),
      "multi_i1": np.random.randint(100, size=(n, 5)),
      "multi_c1": np.random.choice(["x", "y", "z"], size=(n, 5)),
      "multi_b1": np.random.randint(2, size=(n, 5)).astype(np.bool_),
      # Labels
      "label_class_binary": np.random.choice(["l1", "l2"], size=n),
      "label_class_multi": np.random.choice(["l1", "l2", "l3"], size=n),
      "label_regress": np.random.random(size=n),
  }
  return {k: data[k] for k in columns}


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

  def test_feature_encoding_basic(self):
    feature_encoding = to_jax.FeatureEncoding.build(
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
    self.assertIsNotNone(feature_encoding)
    self.assertDictEqual(
        feature_encoding.categorical, {"f2": {"<OOD>": 0, "A": 1, "B": 2}}
    )

  def test_feature_encoding_on_model(self):
    columns = ["f1", "i1", "c1", "b1", "cs1", "label_class_binary"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary",
        num_trees=2,
        features=[("cs1", dataspec_lib.Semantic.CATEGORICAL_SET)],
        include_all_columns=True,
    ).train(create_dataset(columns))
    feature_encoding = to_jax.FeatureEncoding.build(
        model.input_features(), model.data_spec()
    )
    self.assertIsNotNone(feature_encoding)
    self.assertDictEqual(
        feature_encoding.categorical,
        {
            "c1": {"<OOD>": 0, "x": 1, "y": 2, "z": 3},
            "cs1": {"<OOD>": 0, "a": 1, "b": 2, "c": 3},
        },
    )

    encoded_features = feature_encoding.encode(
        {"f1": [1, 2, 3], "c1": ["x", "y", "other"]}
    )
    np.testing.assert_array_equal(
        encoded_features["f1"], jnp.asarray([1, 2, 3])
    )
    np.testing.assert_array_equal(
        encoded_features["c1"], jnp.asarray([1, 2, 0])
    )

  def test_feature_encoding_is_none(self):
    columns = ["f1", "i1", "label_class_binary"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary", num_trees=2
    ).train(create_dataset(columns))
    feature_encoding = to_jax.FeatureEncoding.build(
        model.input_features(), model.data_spec()
    )
    self.assertIsNone(feature_encoding)


class InternalFeatureSpecTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.basic_mapping = to_jax.InternalFeatureSpec([
        generic_model.InputFeature("n1", dataspec_lib.Semantic.NUMERICAL, 0),
        generic_model.InputFeature("n2", dataspec_lib.Semantic.NUMERICAL, 1),
        generic_model.InputFeature("c1", dataspec_lib.Semantic.CATEGORICAL, 2),
        generic_model.InputFeature("c2", dataspec_lib.Semantic.CATEGORICAL, 3),
        generic_model.InputFeature("b1", dataspec_lib.Semantic.BOOLEAN, 4),
        generic_model.InputFeature("b2", dataspec_lib.Semantic.BOOLEAN, 5),
    ])

  def test_basic(self):
    self.assertEqual(self.basic_mapping.numerical, ["n1", "n2"])
    self.assertEqual(self.basic_mapping.categorical, ["c1", "c2"])
    self.assertEqual(self.basic_mapping.boolean, ["b1", "b2"])
    self.assertEqual(self.basic_mapping.inv_numerical, {0: 0, 1: 1})
    self.assertEqual(self.basic_mapping.inv_categorical, {2: 0, 3: 1})
    self.assertEqual(self.basic_mapping.inv_boolean, {4: 0, 5: 1})
    self.assertEqual(
        self.basic_mapping.feature_names, {"n1", "n2", "c1", "c2", "b1", "b2"}
    )

  def test_non_supported_type(self):
    with self.assertRaisesRegex(ValueError, "is not supported"):
      to_jax.InternalFeatureSpec(
          [generic_model.InputFeature("n", dataspec_lib.Semantic.HASH, 0)]
      )

  def test_mapping_convert_empty(self):
    with self.assertRaisesRegex(ValueError, "At least one feature"):
      self.basic_mapping.convert_features({})

  def test_mapping_convert_missing(self):
    with self.assertRaisesRegex(ValueError, "Expecting values with keys"):
      self.basic_mapping.convert_features({"n1": jnp.array([1, 2])})

  def test_mapping_convert_unused(self):
    with self.assertRaisesRegex(ValueError, "Expecting values with keys"):
      self.basic_mapping.convert_features({
          "n1": jnp.array([1, 2]),
          "n2": jnp.array([3, 4]),
          "c1": jnp.array([5, 6]),
          "c2": jnp.array([7, 8]),
          "b1": jnp.array([True, False]),
          "b2": jnp.array([False, True]),
          "other": jnp.array([1, 2]),
      })

  def test_mapping_convert(self):
    internal_values = self.basic_mapping.convert_features({
        "n1": jnp.array([1, 2]),
        "n2": jnp.array([3, 4]),
        "c1": jnp.array([5, 6]),
        "c2": jnp.array([7, 8]),
        "b1": jnp.array([True, False]),
        "b2": jnp.array([False, True]),
    })
    np.testing.assert_array_equal(
        internal_values.numerical, jnp.array([[1.0, 3.0], [2.0, 4.0]])
    )
    np.testing.assert_array_equal(
        internal_values.categorical, jnp.array([[5, 7], [6, 8]])
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
    columns = ["f1", "c1", "f2", "label_regress"]
    model = specialized_learners.RandomForestLearner(
        label="label_regress",
        task=generic_learner.Task.REGRESSION,
        num_trees=1,
    ).train(create_dataset(columns))
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

    self.assertEqual(
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

    self.assertEqual(
        model.get_tree(1).pretty(model.data_spec()),
        """\
'f2' >= 1.5 [score=0 missing=False]
    ├─(pos)─ value=6
    └─(neg)─ value=7
""",
    )

    internal_forest = to_jax.InternalForest(model)

    self.assertEqual(internal_forest.feature_spec.numerical, ["f1", "f2"])
    self.assertEqual(internal_forest.feature_spec.categorical, ["c1"])
    self.assertEqual(internal_forest.feature_spec.boolean, [])
    self.assertEqual(internal_forest.feature_spec.inv_numerical, {1: 0, 3: 1})
    self.assertEqual(internal_forest.feature_spec.inv_categorical, {2: 0})
    self.assertEqual(internal_forest.feature_spec.inv_boolean, {})
    self.assertEqual(
        internal_forest.feature_spec.feature_names, {"f1", "f2", "c1"}
    )

    self.assertEqual(internal_forest.num_trees(), 2)
    self.assertIsNotNone(internal_forest.feature_encoding)

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
    columns = ["f1", "i1", "c1", "label_regress"]
    model = specialized_learners.RandomForestLearner(
        label="label_regress",
        task=generic_learner.Task.REGRESSION,
        num_trees=10,
        max_depth=5,
    ).train(create_dataset(columns))

    internal_forest = to_jax.InternalForest(model)
    self.assertEqual(internal_forest.num_trees(), 10)
    self.assertIsNotNone(internal_forest.feature_encoding)


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
          "label_regress",
          generic_learner.Task.REGRESSION,
          False,
          specialized_learners.GradientBoostedTreesLearner,
      ),
      (
          "gbt_regression_num_cat",
          ["f1", "f2", "c1"],
          "label_regress",
          generic_learner.Task.REGRESSION,
          True,
          specialized_learners.GradientBoostedTreesLearner,
      ),
      (
          "gbt_class_binary_num_cat",
          ["f1", "f2", "c1", "label_class_binary"],
          "label_class_binary",
          generic_learner.Task.CLASSIFICATION,
          True,
          specialized_learners.GradientBoostedTreesLearner,
      ),
      (
          "gbt_class_multi_num_cat",
          ["f1", "f2", "c1", "label_class_multi"],
          "label_class_multi",
          generic_learner.Task.CLASSIFICATION,
          True,
          specialized_learners.GradientBoostedTreesLearner,
      ),
  )
  def test_to_jax_function(
      self,
      features: List[str],
      label: str,
      task: generic_learner.Task,
      has_encoding: bool,
      learner,
  ):

    if learner == specialized_learners.GradientBoostedTreesLearner:
      learner_kwargs = {"validation_ratio": 0.0}
    else:
      learner_kwargs = {}

    # Create YDF model
    columns = features + [label]
    model = learner(
        label=label,
        task=task,
        **learner_kwargs,
    ).train(create_dataset(columns, 1000))

    # Golden predictions
    test_ds = create_dataset(columns, 1000)
    ydf_predictions = model.predict(test_ds)

    # Convert model to tf function
    jax_model = to_jax.to_jax_function(model)
    assert (jax_model.encode is not None) == has_encoding

    # Generate Jax predictions
    del test_ds[label]
    if jax_model.encode is not None:
      input_values = jax_model.encode(test_ds)
    else:
      input_values = {
          k: jnp.asarray(v) for k, v in test_ds.items() if k != label
      }
    jax_predictions = jax_model.predict(input_values)

    # Test predictions
    np.testing.assert_allclose(
        jax_predictions,
        ydf_predictions,
        rtol=1e-5,
        atol=1e-5,
    )

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
        ydf_predictions,
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == "__main__":
  if sys.version_info < (3, 9):
    print("JAX is not supported anymore on python <= 3.8. Skipping JAX tests.")
  else:
    absltest.main()
