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

"""Tests for model learning."""

import os

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import fastavro
import numpy as np
import pandas as pd

# import polars as pl # TODO: Re-enable.

from yggdrasil_decision_forests.learner import abstract_learner_pb2
from ydf.dataset import dataspec
from ydf.learner import generic_learner
from ydf.learner import specialized_learners
from ydf.model import generic_model
from ydf.utils import log

ProtoMonotonicConstraint = abstract_learner_pb2.MonotonicConstraint
Column = dataspec.Column
Semantic = dataspec.Semantic

class LoggingTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 2, False, True)
  def test_logging_function(self, verbose):
    save_verbose = log.verbose(verbose)
    learner = specialized_learners.RandomForestLearner(label="label")
    ds = pd.DataFrame({"feature": [0, 1], "label": [0, 1]})
    _ = learner.train(ds)
    log.verbose(save_verbose)

  @parameterized.parameters(0, 1, 2, False, True)
  def test_logging_arg(self, verbose):
    learner = specialized_learners.RandomForestLearner(label="label")
    ds = pd.DataFrame({"feature": [0, 1], "label": [0, 1]})
    _ = learner.train(ds, verbose=verbose)


class DatasetFormatsTest(parameterized.TestCase):

  def features(self):
    return [
        "f1",
        "f2",
        "i1",
        "i2",
        "c1",
        "multi_c1",
        ("cs1", dataspec.Semantic.CATEGORICAL_SET),
        "multi_f1",
    ]

  # TODO: Re-enable.
  def create_polars_dataset(self, n: int = 1000):
    del n
    raise ValueError("Not available")

  # def create_polars_dataset(self, n: int = 1000) -> pl.DataFrame:
  #   return pl.DataFrame({
  #       # Single-dim features
  #       "f1": np.random.random(size=n),
  #       "f2": np.random.random(size=n),
  #       "i1": np.random.randint(100, size=n),
  #       "i2": np.random.randint(100, size=n),
  #       "c1": np.random.choice(["x", "y", "z"], size=n, p=[0.6, 0.3, 0.1]),
  #       "multi_c1": np.array(
  #           [["a", "x", "z"], ["b", "x", "w"], ["a", "y", "w"],
  #           ["b", "y", "z"]]
  #           * (n // 4)
  #       ),
  #       # Cat-set features
  #       # ================
  #       # Note: Polars as a bug when serializing empty lists of string to Avro
  #       # files (only write one of the two required "optional" bit).
  #       # TODO: Replace [""] by [] once the bug if fixed is added.
  #       "cs1": [["<SOMETHING>"], ["a", "b", "c"], ["b", "c"], ["a"]]
  #         * (n // 4),
  #       # Multi-dim features
  #       # ==================
  #       # Note: It seems support for this type of feature was temporarly dropped
  #       # in Polars 1.9 i.e. the data packing was improved but the avro
  #       # serialization was not implemented. This code would fail with recent
  #       # version of polars with: not yet implemented: write
  #       # FixedSizeList(Field { name: "item", dtype: Float64,
  #         is_nullable: true,
  #       # metadata: {} }, 5) to avro.
  #       "multi_f1": np.random.random(size=(n, 3)),
  #       # # Labels
  #       "label_class_binary1": np.random.choice([False, True], size=n),
  #       "label_class_binary2": np.random.choice([0, 1], size=n),
  #       "label_class_binary3": np.random.choice(["l1", "l2"], size=n),
  #       "label_class_multi1": np.random.choice(["l1", "l2", "l3"], size=n),
  #       "label_class_multi2": np.random.choice([0, 1, 2], size=n),
  #       "label_regress1": np.random.random(size=n),
  #   })

  def test_avro_from_raw_fastavro(self):
    tmp_dir = self.create_tempdir().full_path
    ds_path = os.path.join(tmp_dir, "dataset.avro")
    schema = fastavro.parse_schema({
        "name": "ToyDataset",
        "doc": "A toy dataset.",
        "type": "record",
        "fields": [
            {"name": "f1", "type": "float"},
            {"name": "f2", "type": ["null", "float"]},
            {"name": "i1", "type": "int"},
            {"name": "c1", "type": "string"},
            {
                "name": "multi_f1",
                "type": {"type": "array", "items": "float"},
            },
            {
                "name": "multi_c1",
                "type": {"type": "array", "items": "string"},
            },
            {
                "name": "cs1",
                "type": {"type": "array", "items": "string"},
            },
            {
                "name": "cs2",
                "type": [
                    "null",
                    {"type": "array", "items": ["null", "string"]},
                ],
            },
            {"name": "l", "type": "float"},
        ],
    })
    records = []
    for _ in range(100):
      record = {
          "f1": np.random.rand(),
          "i1": np.random.randint(100),
          "c1": np.random.choice(["x", "y", "z"]),
          "multi_f1": [np.random.rand() for _ in range(3)],
          "multi_c1": [np.random.choice(["x", "y", "z"]) for _ in range(3)],
          "cs1": [
              np.random.choice(["x", "y", "z"])
              for _ in range(np.random.randint(3))
          ],
          "l": np.random.rand(),
      }
      if np.random.rand() < 0.8:
        record["f2"] = np.random.rand()
      if np.random.rand() < 0.8:
        record["cs2"] = [
            np.random.choice(["x", "y", None])
            for _ in range(np.random.randint(3))
        ]
      records.append(record)
    with open(ds_path, "wb") as out:
      fastavro.writer(out, schema, records, codec="deflate")
    learner = specialized_learners.RandomForestLearner(
        label="l",
        num_trees=3,
        features=[
            ("cs1", dataspec.Semantic.CATEGORICAL_SET),
            ("cs2", dataspec.Semantic.CATEGORICAL_SET),
        ],
        include_all_columns=True,
        task=generic_learner.Task.REGRESSION,
    )
    model = learner.train("avro:" + ds_path)
    self.assertEqual(model.num_trees(), 3)
    logging.info("model.input_features():\n%s", model.input_features())
    InputFeature = generic_model.InputFeature
    Semantic = dataspec.Semantic
    self.assertEqual(
        model.input_features(),
        [
            InputFeature(name="f1", semantic=Semantic.NUMERICAL, column_idx=0),
            InputFeature(name="f2", semantic=Semantic.NUMERICAL, column_idx=1),
            InputFeature(name="i1", semantic=Semantic.NUMERICAL, column_idx=2),
            InputFeature(
                name="c1", semantic=Semantic.CATEGORICAL, column_idx=3
            ),
            InputFeature(
                name="cs1", semantic=Semantic.CATEGORICAL_SET, column_idx=4
            ),
            InputFeature(
                name="cs2", semantic=Semantic.CATEGORICAL_SET, column_idx=5
            ),
            InputFeature(
                name="multi_f1.0_of_3",
                semantic=Semantic.NUMERICAL,
                column_idx=7,
            ),
            InputFeature(
                name="multi_f1.1_of_3",
                semantic=Semantic.NUMERICAL,
                column_idx=8,
            ),
            InputFeature(
                name="multi_f1.2_of_3",
                semantic=Semantic.NUMERICAL,
                column_idx=9,
            ),
            InputFeature(
                name="multi_c1.0_of_3",
                semantic=Semantic.CATEGORICAL,
                column_idx=10,
            ),
            InputFeature(
                name="multi_c1.1_of_3",
                semantic=Semantic.CATEGORICAL,
                column_idx=11,
            ),
            InputFeature(
                name="multi_c1.2_of_3",
                semantic=Semantic.CATEGORICAL,
                column_idx=12,
            ),
        ],
    )

  def test_avro_from_fastavro_with_pandas(self):
    tmp_dir = self.create_tempdir().full_path
    ds_path = os.path.join(tmp_dir, "dataset.avro")
    schema = fastavro.parse_schema({
        "name": "ToyDataset",
        "doc": "A toy dataset.",
        "type": "record",
        "fields": [
            {"name": "f1", "type": "float"},
            {"name": "f2", "type": "float"},
            {"name": "f3", "type": ["null", "float"]},
            {"name": "l", "type": "float"},
        ],
    })
    ds = pd.DataFrame({
        "f1": np.random.rand(100),
        "f2": np.random.rand(100),
        "f3": np.random.rand(100),
        "l": np.random.rand(100),
    })
    with open(ds_path, "wb") as out:
      fastavro.writer(out, schema, ds.to_dict("records"), codec="deflate")
    learner = specialized_learners.RandomForestLearner(
        label="l",
        num_trees=3,
        task=generic_learner.Task.REGRESSION,
    )
    model = learner.train("avro:" + ds_path)
    self.assertEqual(model.num_trees(), 3)


class EmptyListFeatureTest(parameterized.TestCase):

  @parameterized.parameters([
      specialized_learners.GradientBoostedTreesLearner(label="l"),
      specialized_learners.RandomForestLearner(label="l"),
      specialized_learners.IsolationForestLearner(),
      specialized_learners.CartLearner(label="l"),
  ])
  def test_empty_feature_default_semantic(self, learner):
    ds = {"l": [1, 0], "f": np.array([[], []], dtype=object)}
    with self.assertRaisesRegex(
        ValueError,
        "Feature 'f' is detected as a multi-dimensional fixed-length feature"
        " with dimension zero",
    ):
      _ = learner.train(ds)

  def test_empty_feature_catset_nonempty_prediction(self):
    ds_train = {"l": [1, 0], "f": np.array([[], []], dtype=object)}
    ds_test = {"f": np.array([["a"], []], dtype=object)}
    # Note: Random Forests with bootstrapping might have unbalanced predictions.
    model = specialized_learners.GradientBoostedTreesLearner(
        label="l", num_trees=1, features=[("f", Semantic.CATEGORICAL_SET)]
    ).train(ds_train)
    np.testing.assert_equal(model.predict(ds_test), np.array([0.5, 0.5]))

  def test_empty_feature_catset_empty_prediction(self):
    ds_train = {"l": [1, 0], "f": np.array([[], []], dtype=object)}
    ds_test = {"f": np.array([[], []], dtype=object)}
    # Note: Random Forests with bootstrapping might have unbalanced predictions.
    model = specialized_learners.GradientBoostedTreesLearner(
        label="l", num_trees=1, features=[("f", Semantic.CATEGORICAL_SET)]
    ).train(ds_train)
    np.testing.assert_equal(model.predict(ds_test), np.array([0.5, 0.5]))

  def test_nonempty_feature_catset_empty_prediction(self):
    ds_train = {"l": [1, 0], "f": np.array([["a"], ["b"]], dtype=object)}
    ds_test = {"f": np.array([[], []], dtype=object)}
    # Note: Random Forests with bootstrapping might have unbalanced predictions.
    model = specialized_learners.GradientBoostedTreesLearner(
        label="l",
        num_trees=1,
        features=[("f", Semantic.CATEGORICAL_SET)],
        min_vocab_frequency=1,
        min_examples=1,
    ).train(ds_train)
    np.testing.assert_equal(model.predict(ds_test), np.array([0.5, 0.5]))

  def test_nonempty_feature_default_multidim_empty_prediction(self):
    ds_train = {"l": [1, 0], "f": np.array([["a"], ["b"]], dtype=object)}
    ds_test = {"f": np.array([[], []], dtype=object)}
    # Note: Random Forests with bootstrapping might have unbalanced predictions.
    model = specialized_learners.GradientBoostedTreesLearner(
        label="l",
        num_trees=1,
        min_vocab_frequency=1,
        min_examples=1,
    ).train(ds_train)
    with self.assertRaisesRegex(
        ValueError,
        "Feature 'f' is expected as a multi-dimensional feature of dimension 1"
        " and type CATEGORICAL.",
    ):
      _ = model.predict(ds_test)

  def test_nonempty_feature_default_catset_empty_prediction(self):
    ds_train = {"l": [1, 0], "f": np.array([["a"], ["b", "c"]], dtype=object)}
    ds_test = {"f": np.array([[], []], dtype=object)}
    # Note: Random Forests with bootstrapping might have unbalanced predictions.
    model = specialized_learners.GradientBoostedTreesLearner(
        label="l",
        num_trees=1,
        min_vocab_frequency=1,
        min_examples=1,
    ).train(ds_train)
    np.testing.assert_equal(model.predict(ds_test), np.array([0.5, 0.5]))


class UtilityTest(absltest.TestCase):

  def test_feature_name_to_regex(self):
    self.assertEqual(
        generic_learner._feature_name_to_regex("a(z)e"), r"^a\(z\)e$"
    )


if __name__ == "__main__":
  absltest.main()
