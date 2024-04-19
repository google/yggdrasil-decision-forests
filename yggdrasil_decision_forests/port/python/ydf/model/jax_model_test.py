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

from typing import Any, Dict, List
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.dataset import dataspec as dataspec_lib
from ydf.learner import specialized_learners
from ydf.model import export_jax as to_jax
from ydf.model import generic_model


class JaxModelTest(parameterized.TestCase):

  def create_dataset(self, columns: List[str]) -> Dict[str, Any]:
    """Creates a dataset with random values."""
    data = {
        # Single-dim features
        "f1": np.random.random(size=100),
        "f2": np.random.random(size=100),
        "i1": np.random.randint(100, size=100),
        "i2": np.random.randint(100, size=100),
        "c1": np.random.choice(["x", "y", "z"], size=100, p=[0.6, 0.3, 0.1]),
        "b1": np.random.randint(2, size=100).astype(np.bool_),
        "b2": np.random.randint(2, size=100).astype(np.bool_),
        # Cat-set features
        "cs1": [[], ["a", "b", "c"], ["b", "c"], ["a"]] * 25,
        # Multi-dim features
        "multi_f1": np.random.random(size=(100, 5)),
        "multi_f2": np.random.random(size=(100, 5)),
        "multi_i1": np.random.randint(100, size=(100, 5)),
        "multi_c1": np.random.choice(["x", "y", "z"], size=(100, 5)),
        "multi_b1": np.random.randint(2, size=(100, 5)).astype(np.bool_),
        # Labels
        "label_class_binary": np.random.choice(["l1", "l2"], size=100),
        "label_class_multi": np.random.choice(["l1", "l2", "l3"], size=100),
        "label_regress": np.random.random(size=100),
    }
    return {k: data[k] for k in columns}

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
    ).train(self.create_dataset(columns))
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
        encoded_features["f1"], jax.numpy.asarray([1, 2, 3])
    )
    np.testing.assert_array_equal(
        encoded_features["c1"], jax.numpy.asarray([1, 2, 0])
    )

  def test_feature_encoding_is_none(self):
    columns = ["f1", "i1", "label_class_binary"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary", num_trees=2
    ).train(self.create_dataset(columns))
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



if __name__ == "__main__":
  absltest.main()
