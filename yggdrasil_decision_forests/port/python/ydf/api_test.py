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

"""Test the API of YDF."""

import math
import os
import pickle
import sys

from absl import logging
from absl.testing import absltest
import fastavro
from google.protobuf import text_format
import numpy as np
import pandas as pd
from sklearn import ensemble as skl_ensemble
import sklearn.datasets

import ydf  # In the world, use "import ydf"
from ydf.utils import test_utils


class ApiTest(absltest.TestCase):

  def test_create_dataset(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, math.nan],
        "c2": [1, 2, 3],
        "c3": [True, False, True],
        "c4": ["x", "y", ""],
    })
    ds = ydf.create_vertical_dataset(pd_ds)
    logging.info("Dataset:\n%s", ds)

  def test_create_dataset_with_column(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, math.nan],
        "c2": [1, 2, 3],
        "c3": [True, False, True],
        "c4": ["x", "y", ""],
    })
    ds = ydf.create_vertical_dataset(
        pd_ds,
        columns=[
            "c1",
            ("c2", ydf.Semantic.NUMERICAL),
            ydf.Column("c3"),
            ydf.Column("c4", ydf.Semantic.CATEGORICAL),
        ],
    )
    logging.info("Dataset:\n%s", ds)

  def test_load_and_save_model(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    model = ydf.load_model(model_path)
    logging.info(model)
    tempdir = self.create_tempdir().full_path
    model.save(tempdir, ydf.ModelIOOptions(file_prefix="model_prefix_"))
    logging.info(os.listdir(tempdir))

  def test_serialize_and_deserialize_model(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    model = ydf.load_model(model_path)
    serialized_model = model.serialize()
    deserialized_model = ydf.deserialize_model(serialized_model)
    logging.info(deserialized_model)

  def test_save_model_with_different_node_formats(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    model = ydf.load_model(model_path)
    serialized_model = model.serialize()

    logging.info("serialized_model size: %s", len(serialized_model))
    tmp_dir = self.create_tempdir().full_path

    model.set_node_format(ydf.NodeFormat.BLOB_SEQUENCE)
    model.save(os.path.join(tmp_dir, "blob"))

    model.set_node_format(ydf.NodeFormat.BLOB_SEQUENCE_GZIP)
    model.save(os.path.join(tmp_dir, "blob_gzip"))

  def test_pickle_and_unpickle_model(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    model = ydf.load_model(model_path)
    pickled_model = pickle.dumps(model)
    unpickled_model = pickle.loads(pickled_model)
    logging.info(unpickled_model)

  def test_train_random_forest(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    model = ydf.RandomForestLearner(num_trees=3, label="label").train(pd_ds)
    logging.info(model)

  def test_train_gradient_boosted_tree(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    model = ydf.GradientBoostedTreesLearner(num_trees=10, label="label").train(
        pd_ds
    )
    logging.info(model)

  def test_train_cart(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    model = ydf.CartLearner(label="label").train(pd_ds)
    logging.info(model)

  def test_evaluate_model(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = ydf.load_model(model_path)
    test_ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        )
    )
    evaluation = model.evaluate(test_ds)
    logging.info("Evaluation:\n%s", evaluation)
    self.assertAlmostEqual(evaluation.accuracy, 0.87235, 3)

    tempdir = self.create_tempdir().full_path
    with open(os.path.join(tempdir, "evaluation.html"), "w") as f:
      f.write(evaluation.html())

  def test_analyze_model(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = ydf.load_model(model_path)
    test_ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        )
    )
    analysis = model.analyze(test_ds)
    logging.info("Analysis:\n%s", analysis)

    tempdir = self.create_tempdir().full_path
    with open(os.path.join(tempdir, "analysis.html"), "w") as f:
      f.write(analysis.html())

  def test_cross_validation(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    learner = ydf.RandomForestLearner(num_trees=3, label="label")
    evaluation = learner.cross_validation(pd_ds)
    logging.info(evaluation)

  def test_export_to_cc(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = ydf.load_model(model_path)
    logging.info(
        "Copy the following in a .h file to run the model in C++:\n%s",
        model.to_cpp(),
    )

  def test_verbose_full(self):
    save_verbose = ydf.verbose(2)
    learner = ydf.RandomForestLearner(label="label")
    _ = learner.train(pd.DataFrame({"feature": [0, 1], "label": [0, 1]}))
    ydf.verbose(save_verbose)

  def test_verbose_default(self):
    learner = ydf.RandomForestLearner(label="label")
    _ = learner.train(pd.DataFrame({"feature": [0, 1], "label": [0, 1]}))

  def test_verbose_none(self):
    save_verbose = ydf.verbose(0)
    learner = ydf.RandomForestLearner(label="label")
    _ = learner.train(pd.DataFrame({"feature": [0, 1], "label": [0, 1]}))
    ydf.verbose(save_verbose)

  def test_print_a_tree(self):
    train_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    learner = ydf.CartLearner(label="label")
    model = learner.train(train_ds)
    assert isinstance(model, ydf.CARTModel)
    model.print_tree(tree_idx=0)

  def test_get_a_tree(self):
    train_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    learner = ydf.RandomForestLearner(label="label")
    model = learner.train(train_ds)
    assert isinstance(model, ydf.RandomForestModel)
    tree = model.get_tree(tree_idx=0)
    logging.info("Found tree:\n%s", tree)

  def test_list_input_features(self):
    train_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    learner = ydf.RandomForestLearner(label="label")
    model = learner.train(train_ds)
    logging.info("Input features:\n%s", model.input_features())

  def test_export_tensorflow_saved_model(self):
    if sys.version_info < (3, 9):
      print(
          "TFDF is not supported anymore on python <= 3.8. Skipping TFDF tests."
      )
      return
    if not sys.version_info < (3, 12):
      print(
          "TFDF is not yet supported for python >= 3.12. Skipping TFDF tests."
      )
      return

    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    model = ydf.load_model(model_path)
    tempdir = self.create_tempdir().full_path
    model.to_tensorflow_saved_model(tempdir, mode="tf")

  def test_export_jax_function(self):
    if sys.version_info < (3, 9):
      print(
          "JAX is not supported anymore on python <= 3.8. Skipping JAX tests."
      )
      return

    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = ydf.load_model(model_path)
    _ = model.to_jax_function()

  def test_import_sklearn_model(self):
    X, y = sklearn.datasets.make_classification()
    skl_model = skl_ensemble.RandomForestClassifier().fit(X, y)
    ydf_model = ydf.from_sklearn(skl_model)
    _ = ydf_model.predict({"features": X})

  def test_read_tf_record(self):
    ds_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_train.recordio.gz"
    )

    def process(example):
      for key in ["workclass", "occupation", "native_country"]:
        if key not in example.features.feature:
          example.features.feature[key].bytes_list.value.append(b"")
      return example

    ds = ydf.util.read_tf_record(ds_path, process=process)
    logging.info("%s", ds)

  def test_io_benchmark(self):
    tempdir = self.create_tempdir().full_path
    ydf.internal.benchmark_io_speed(
        work_dir=tempdir,
        num_examples=100,
        num_features=5,
        num_shards=2,
        create_workdir=False,
    )

  def test_train_on_avro_files(self):
    # TODO: Use Polars's Avro writer instead.
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

    ds_path = os.path.join(self.create_tempdir().full_path, "ds.avro")
    with open(ds_path, "wb") as out:
      fastavro.writer(out, schema, ds.to_dict("records"), codec="deflate")

    learner = ydf.RandomForestLearner(label="l", task=ydf.Task.REGRESSION)
    model = learner.train("avro:" + ds_path)
    logging.info(model)

  def test_train_config_proto(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })

    extra_training_config = text_format.Parse(
        """
[yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
  num_trees: 10
  decision_tree {
    max_depth: 5
  }
}
""",
        ydf.TrainingConfig(),
    )

    model = ydf.RandomForestLearner(
        label="label",
        min_examples=2,
        num_trees=20,  # Ignored as "num_trees" is set in the train config.
        extra_training_config=extra_training_config,
    ).train(pd_ds)
    logging.info(model)
    self.assertEqual(model.num_trees(), 10)

  def test_help(self):
    ydf.help.loading_data()


if __name__ == "__main__":
  absltest.main()
