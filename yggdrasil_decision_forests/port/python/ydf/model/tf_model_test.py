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

"""Model tests depending on TensorFlow.

TensorFlow cannot be compiled in debug mode, so these tests are separated out to
improve debuggability of the remaining model tests.
"""

import math
import os
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

from ydf.dataset import dataspec
from ydf.learner import generic_learner
from ydf.learner import specialized_learners
from ydf.model import export_tf
from ydf.model import model_lib
from ydf.utils import test_utils


class TfModelTest(parameterized.TestCase):

  def create_ds(
      self, columns: List[str], label: Optional[str], weights: Optional[str]
  ) -> pd.DataFrame:
    df = pd.DataFrame({})
    if "cat_int_small" in columns:
      df["cat_int_small"] = [1, 2, 3, 4, 3, 2, 1, 0, -1, -2]
    if "cat_int_large" in columns:
      df["cat_int_large"] = [100, 200, 300, 400, 300, 20, 100, 0, -100, -200]
    if "cat_string" in columns:
      df["cat_string"] = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d"]
    if "num" in columns:
      df["num"] = [0, 1, 2, 3, 4, 5.5, 6.6, 7.7, 8.8, 9.9]

    if weights is not None:
      df["weights"] = list(range(1, 11))

    if label == "classification_binary_int":
      df["label"] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    elif label == "classification_binary_str":
      df["label"] = ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"]
    elif label == "classification_multiclass_int":
      df["label"] = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    elif label == "classification_multiclass_str":
      df["label"] = ["a", "b", "c", "d", "a", "b", "c", "d", "a", "b"]
    elif label == "regression":
      df["label"] = [42.5, 43.3, 73.4, 21.1, 26.4, 9.3, -1, -44, -23.4, 234.3]
    elif label == "ranking":
      df["ranking_group"] = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]
      df["label"] = [0, 0, 0, 1, 1, 1, 3, 3, 4, 4]
    return df

  def create_dataset_v2(self, columns: List[str]) -> Dict[str, Any]:
    """Creates a dataset with random values."""
    data = {
        # Single-dim features
        "f1": np.random.random(size=100),
        "f2": np.random.random(size=100),
        "i1": np.random.randint(100, size=100),
        "i2": np.random.randint(100, size=100),
        "c1": np.random.choice(["x", "y", "z"], size=100),
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

  def create_csv(
      self, columns: List[str], label: Optional[str], weights: Optional[str]
  ) -> str:
    tmp_dir = self.create_tempdir()
    csv_file = tmp_dir.create_file("file.csv")
    self.create_ds(columns, label, weights).to_csv(
        csv_file.full_path, index=False
    )
    return csv_file.full_path

  def create_tfdf_model_and_test_df(
      self,
      ds_type: str,
      feature_col: str,
      label_col: str,
      weight_col: str,
      model_path: str,
  ) -> Tuple[tfdf.keras.GradientBoostedTreesModel, pd.DataFrame]:
    # Set feature-specific hyperparameters of the model.
    if feature_col.startswith("cat_"):
      column_semantic = tfdf.keras.FeatureSemantic.CATEGORICAL
    elif feature_col.startswith("num"):
      column_semantic = tfdf.keras.FeatureSemantic.NUMERICAL
    else:
      raise ValueError(f"Could not determine semantic of column {feature_col}")
    needs_min_vocab_frequency = (
        column_semantic == tfdf.keras.FeatureSemantic.CATEGORICAL
    )

    # Set task-specific hyperparameters of the model.
    ranking_group = None
    if label_col.startswith("classification"):
      task = tfdf.keras.Task.CLASSIFICATION
    elif label_col.startswith("regression"):
      task = tfdf.keras.Task.REGRESSION
    elif label_col.startswith("ranking"):
      task = tfdf.keras.Task.RANKING
      ranking_group = "ranking_group"
    else:
      raise ValueError(f"Could not determine task for label {label_col}")

    # Create an empty, small TF-DF model with the given hyperparameters.
    tfdf_model = tfdf.keras.GradientBoostedTreesModel(
        task=task,
        min_examples=1,
        num_trees=5,
        ranking_group=ranking_group,
        features=[
            tfdf.keras.FeatureUsage(
                name=feature_col,
                semantic=column_semantic,
                min_vocab_frequency=2 if needs_min_vocab_frequency else None,
            )
        ],
    )

    # Create the training dataset and fit the model to it.
    if ds_type == "pd":
      train_df = self.create_ds(
          columns=[feature_col], label=label_col, weights=weight_col
      )
      train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
          train_df, label="label", task=task, weight="weights"
      )
      tfdf_model.fit(train_ds)
    elif ds_type == "file":
      train_ds_path = self.create_csv(
          columns=[feature_col], label=label_col, weights=weight_col
      )
      tfdf_model.fit_on_dataset_path(
          train_ds_path, label_key="label", weight_key="weights"
      )
    else:
      raise ValueError(f"Unknown dataset type {ds_type}")
    tfdf_model.save(model_path)

    # Create the test dataset. Note that the test dataset is always a Pandas
    # dataframe, since predicting from file is not supported.
    test_df = self.create_ds(columns=[feature_col], label=None, weights=None)
    return tfdf_model, test_df

  @parameterized.product(
      feature_col=["cat_int_small", "cat_int_large", "cat_string", "num"],
      label_col=[
          "classification_binary_int",
          "classification_binary_str",
          "classification_multiclass_int",
          "classification_multiclass_str",
          "regression",
          "ranking",
      ],
      ds_type=["file", "pd"],
  )
  def test_tfdf_ydf_prediction_equality(
      self, feature_col: str, label_col: str, ds_type: str
  ):
    model_dir = self.create_tempdir().full_path
    # When reading from file, TF-DF casts integer categories to strings, but it
    # doesn't when converting from Pandas, so the only way to feed matching data
    # to the model is to just feed it string data, and we omit the int cases.
    if ds_type == "file" and feature_col.startswith("cat_int"):
      self.skipTest(
          "Categorical Integer features don't work in TF-DF when reading from"
          " file"
      )

    tfdf_model, test_df = self.create_tfdf_model_and_test_df(
        ds_type, feature_col, label_col, "weights", model_dir
    )
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

    tfdf_predictions = tfdf_model.predict(test_ds)
    ydf_model = model_lib.from_tensorflow_decision_forests(model_dir)
    ydf_predictions = ydf_model.predict(test_df)
    npt.assert_allclose(
        tfdf_predictions.flatten(), ydf_predictions.flatten(), atol=0.0001
    )

  @parameterized.product(
      feature_col=["cat_string", "num"],
      label_col=[
          "classification_binary_int",
          "classification_binary_str",
          "classification_multiclass_int",
          "classification_multiclass_str",
          "regression",
          "ranking",
      ],
      ds_type=["file", "pd"],
  )
  def test_tfdf_convert_back_and_forth(
      self, feature_col: str, label_col: str, ds_type: str
  ):
    model_dir = self.create_tempdir().full_path

    tfdf_model, test_df = self.create_tfdf_model_and_test_df(
        ds_type, feature_col, label_col, "weights", model_dir
    )
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)
    new_model_dir = self.create_tempdir().full_path

    # Create the YDF model.
    ydf_model = model_lib.from_tensorflow_decision_forests(model_dir)
    # Convert the YDF model back to a TF-DF model and load it.
    ydf_model.to_tensorflow_saved_model(new_model_dir)
    new_tfdf_model = tf.keras.models.load_model(new_model_dir)

    # Check for prediction equality.
    tfdf_predictions = tfdf_model.predict(test_ds)
    new_tfdf_predictions = new_tfdf_model.predict(test_ds)
    npt.assert_allclose(
        tfdf_predictions.flatten(), new_tfdf_predictions.flatten(), atol=0.0001
    )

  @parameterized.product(
      feature_col=["cat_int_small", "cat_int_large"],
      label_col=[
          "classification_binary_int",
          "classification_binary_str",
          "classification_multiclass_int",
          "classification_multiclass_str",
          "regression",
          "ranking",
      ],
      ds_type=["pd"],
  )
  def test_tfdf_convert_back_and_forth_cat_int(
      self, feature_col: str, label_col: str, ds_type: str
  ):
    model_dir = self.create_tempdir().full_path

    tfdf_model, test_df = self.create_tfdf_model_and_test_df(
        ds_type, feature_col, label_col, "weights", model_dir
    )
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)
    new_model_dir = self.create_tempdir().full_path

    # Create the YDF model.
    ydf_model = model_lib.from_tensorflow_decision_forests(model_dir)
    # Convert the YDF model back to a TF-DF model and load it.
    ydf_model.to_tensorflow_saved_model(new_model_dir)
    new_tfdf_model = tf.keras.models.load_model(new_model_dir)

    # Prepare the test dataset for the loaded model: Categorical integer
    # features are now string features.
    new_test_df = test_df.copy(deep=True)
    new_test_df[feature_col] = new_test_df.astype(np.str_)
    new_test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(new_test_df)

    # Check for prediction equality.
    tfdf_predictions = tfdf_model.predict(test_ds)
    new_tfdf_predictions = new_tfdf_model.predict(new_test_ds)
    npt.assert_allclose(
        tfdf_predictions.flatten(), new_tfdf_predictions.flatten(), atol=0.0001
    )

  def test_to_tensorflow_saved_model_serialized_input(self):
    # TODO: b/321204507 - Integrate logic in YDF.

    train_ds_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_train.recordio"
    )
    test_ds_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.recordio"
    )

    # Train a model on the tfrecord directly.
    ydf_model = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        num_trees=10,
    ).train(f"tfrecordv2+tfe:{train_ds_path}")

    test_predictions = ydf_model.predict(f"tfrecordv2+tfe:{test_ds_path}")

    tempdir = self.create_tempdir().full_path

    # Export the model to a TensorFlow Saved Model.
    #
    # This model expects inputs in the form of a dictionary of features
    # values e.g. {"age": [...], "capital_gain": [...], ...}.
    #
    # This is referred as the "predict" API.
    path_wo_signature = os.path.join(tempdir, "mdl")
    ydf_model.to_tensorflow_saved_model(path_wo_signature)

    # Load the model, and add a serialized tensorflow examples protobuffer
    # input signature. In other words, the model expects as input a serialized
    # tensorflow example proto.
    #
    # This is often referred the "classify" or "regress" API.
    path_w_signature = os.path.join(tempdir, "mdl_ws")
    tf_model_wo_signature = tf.keras.models.load_model(path_wo_signature)

    # The list of input features to read from the tensorflow example proto.
    # Note that tensorflow example proto dtypes can only be float32, int64 or
    # string.
    feature_spec = {
        # Numerical
        "age": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "capital_gain": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "capital_loss": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "education_num": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "fnlwgt": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "hours_per_week": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        # Categorical
        "education": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "marital_status": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "native_country": tf.io.FixedLenFeature(
            shape=[], dtype=tf.string, default_value=""
        ),
        "occupation": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "race": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "relationship": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "sex": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "workclass": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    }

    # The "classify" requires for all the predictions to return the label
    # classes. This only make sense for a classification model.
    label_classes = ydf_model.label_classes()

    # The "make_classify_fn" function combines a tensorflow example proto
    # parsing stage with the classical model inference.
    def make_classify_fn(
        model: tf.keras.Model,
    ) -> Callable[[tf.Tensor], Mapping[str, tf.Tensor]]:
      @tf.function(
          input_signature=[
              tf.TensorSpec([None], dtype=tf.string, name="inputs")
          ]
      )
      def classify(
          serialized_tf_examples: tf.Tensor,
      ) -> Mapping[str, tf.Tensor]:
        # Parse the serialized tensorflow proto examples into a dictionary of
        # tensor values.
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        # Cast all the int64 features into float32 values.
        #
        # The SavedModel exported by YDF expects float32 value for all
        # numerical features (can be overridden with the
        # `input_model_signature_fn` argument). However, in this dataset, the
        # numerical features are stored as int64.
        for feature in parsed_features:
          if parsed_features[feature].dtype == tf.int64:
            parsed_features[feature] = tf.cast(
                parsed_features[feature], tf.float32
            )

        # Apply the model.
        outputs = model(parsed_features)

        # Extract the label classes. The "classify" API expects for the label
        # classes to be returned with every predictions.
        batch_size = tf.shape(serialized_tf_examples)[0]
        batched_label_classes = tf.broadcast_to(
            input=tf.constant(label_classes),
            shape=(batch_size, len(label_classes)),
        )

        return {"classes": batched_label_classes, "scores": outputs}

      return classify

    # Save the model to a SavedModel with the "classify" signature.
    signatures = {
        "classify": make_classify_fn(tf_model_wo_signature),
    }
    tf.saved_model.save(
        tf_model_wo_signature, path_w_signature, signatures=signatures
    )
    tf_model_w_signature = tf.saved_model.load(path_w_signature)

    logging.info("Available signatures: %s", tf_model_w_signature.signatures)

    # Generate predictions with the "classify" signature.
    classify_model = tf_model_w_signature.signatures["classify"]

    for example_idx, serialized_example in enumerate(
        tf.data.TFRecordDataset([test_ds_path]).take(10)
    ):
      prediction = classify_model(inputs=[serialized_example.numpy()])
      logging.info("prediction:%s", prediction)
      npt.assert_almost_equal(
          prediction["scores"].numpy(), test_predictions[example_idx]
      )
      npt.assert_equal(prediction["classes"].numpy(), [[b"<=50K", b">50K"]])

  def test_to_tensorflow_function(self):
    """A simple function conversion."""

    # Create YDF model
    columns = ["f1", "f2", "i1", "c1", "b1", "cs1", "label_class_binary"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary",
        num_trees=10,
        features=[("cs1", dataspec.Semantic.CATEGORICAL_SET)],
        include_all_columns=True,
    ).train(self.create_dataset_v2(columns))

    # Golden predictions
    test_ds = self.create_dataset_v2(columns)
    ydf_predictions = model.predict(test_ds)

    # Convert model to tf function + generate predictions
    tf_function = model.to_tensorflow_function()
    tf_test_ds = {
        "f1": tf.constant(test_ds["f1"]),
        "f2": tf.constant(test_ds["f2"]),
        "i1": tf.constant(test_ds["i1"]),
        "c1": tf.constant(test_ds["c1"]),
        "b1": tf.constant(test_ds["b1"]),
        "cs1": tf.ragged.constant(test_ds["cs1"]),
    }
    tf_predictions = tf_function(tf_test_ds)

    npt.assert_array_equal(ydf_predictions, tf_predictions)

  @parameterized.product(can_be_saved=[True, False])
  def test_to_multi_tensorflow_function(self, can_be_saved: bool):
    """Function export and serialization with multiple YDF models."""

    # Create YDF model
    columns = ["f1", "f2", "i1", "c1", "label_class_binary"]
    model_1 = specialized_learners.RandomForestLearner(
        label="label_class_binary", num_trees=10
    ).train(self.create_dataset_v2(columns))
    model_2 = specialized_learners.RandomForestLearner(
        label="label_class_binary", num_trees=10
    ).train(self.create_dataset_v2(columns))

    # Golden predictions
    test_ds = self.create_dataset_v2(columns)
    ydf_predictions = model_1.predict(test_ds) + model_2.predict(test_ds) * 2

    # Convert to tf function + generate predictions
    tf_function_m1 = model_1.to_tensorflow_function(can_be_saved=can_be_saved)
    tf_function_m2 = model_2.to_tensorflow_function(can_be_saved=can_be_saved)

    @tf.function
    def tf_function(features):
      return tf_function_m1(features) + tf_function_m2(features) * 2

    tf_test_ds = {k: tf.constant(v) for k, v in test_ds.items()}
    tf_predictions = tf_function(tf_test_ds)

    # Check predictions
    npt.assert_array_equal(ydf_predictions, tf_predictions)

    if not can_be_saved:
      return

    # Serialize / unserialize model
    with tempfile.TemporaryDirectory() as tmp_dir:
      tf_model = tf.Module()
      tf_model.__call__ = tf_function
      tf_model.tf_function_m1 = tf_function_m1
      tf_model.tf_function_m2 = tf_function_m2
      tf.saved_model.save(tf_model, tmp_dir)
      loaded_tf_model = tf.saved_model.load(tmp_dir)

    # Check predictions gain
    loaded_tf_predictions = loaded_tf_model(tf_test_ds)
    npt.assert_array_equal(ydf_predictions, loaded_tf_predictions)

  def test_to_tensorflow_function_with_multidim_input(self):
    # Train a model
    columns = [
        "multi_f1",
        "multi_i1",
        "multi_c1",
        "multi_b1",
        "label_class_binary",
    ]
    train_ds = self.create_dataset_v2(columns)
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary"
    ).train(train_ds)
    # Generate predictions with YDF
    ydf_predictions = model.predict(train_ds)

    # Convert ydf model into a tf function
    tf_function = model.to_tensorflow_function()

    # Validate the tf model predictions
    tf_test_ds = {k: tf.constant(train_ds[k]) for k in columns[:-1]}
    tf_predictions = tf_function(tf_test_ds)
    npt.assert_array_equal(ydf_predictions, tf_predictions)

  def test_to_raw_tensorflow_saved_model(self):
    """Simple export to SavedModel format."""

    # Create YDF model
    columns = ["f1", "f2", "i1", "c1", "b1", "cs1", "label_class_binary"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary",
        num_trees=10,
        features=[("cs1", dataspec.Semantic.CATEGORICAL_SET)],
        include_all_columns=True,
    ).train(self.create_dataset_v2(columns))

    # Golden predictions
    test_ds = self.create_dataset_v2(columns[:-1])
    ydf_predictions = model.predict(test_ds)

    # Save model
    tmp_dir = self.create_tempdir().full_path
    model.to_tensorflow_saved_model(
        tmp_dir,
        mode="tf",
        servo_api=False,
        feed_example_proto=False,
        feature_dtypes={"f1": tf.float32},
    )

    # Load model
    tf_model = tf.saved_model.load(tmp_dir)

    # Test predictions
    tf_test_ds = {
        # While f1 was feed as a float64, it was saved as a float32.
        "f1": tf.constant(test_ds["f1"], tf.float32),
        "f2": tf.constant(test_ds["f2"]),
        "i1": tf.constant(test_ds["i1"]),
        "c1": tf.constant(test_ds["c1"]),
        "b1": tf.constant(test_ds["b1"]),
        "cs1": tf.ragged.constant(test_ds["cs1"]),
    }
    tf_predictions = tf_model(tf_test_ds)
    npt.assert_equal(ydf_predictions, tf_predictions)

  @parameterized.parameters(True, False)
  def test_to_raw_tensorflow_saved_model_with_multidim_input(
      self, with_filter: bool
  ):
    # Create YDF model
    columns = [
        "multi_f1",
        "multi_i1",
        "multi_c1",
        "multi_b1",
        "label_class_binary",
    ]
    feature_columns = columns[:-2] if with_filter else columns[:-1]
    train_ds = self.create_dataset_v2(columns)
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary",
        num_trees=10,
        features=feature_columns if with_filter else None,
    ).train(train_ds)

    # Golden predictions
    ydf_predictions = model.predict(train_ds)

    # Save model
    tmp_dir = self.create_tempdir().full_path
    model.to_tensorflow_saved_model(
        tmp_dir,
        mode="tf",
        servo_api=False,
        feed_example_proto=False,
    )

    # Load model
    tf_model = tf.saved_model.load(tmp_dir)

    # Test predictions
    tf_test_ds = {k: tf.constant(train_ds[k]) for k in feature_columns}
    tf_predictions = tf_model(tf_test_ds)
    npt.assert_equal(ydf_predictions, tf_predictions)

  def test_to_tensorflow_saved_model_classify_api(self):
    """Export to SavedModel format with regress API."""
    columns = ["f1", "f2", "label_class_multi"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_multi",
        num_trees=10,
        task=generic_learner.Task.CLASSIFICATION,
    ).train(self.create_dataset_v2(columns))
    test_ds = self.create_dataset_v2(columns[:-1])
    ydf_predictions = model.predict(test_ds)
    tmp_dir = self.create_tempdir().full_path
    model.to_tensorflow_saved_model(
        path=tmp_dir,
        mode="tf",
        feed_example_proto=False,
        servo_api=True,
    )
    tf_model = tf.saved_model.load(tmp_dir)
    tf_model_predict = tf_model.signatures["serving_default"]
    tf_prediction = tf_model_predict(**test_ds)
    self.assertEqual(tf_prediction["scores"].shape, (100, 3))
    self.assertEqual(tf_prediction["classes"].shape, (100, 3))
    npt.assert_equal(ydf_predictions, tf_prediction["scores"])

  def test_to_tensorflow_saved_model_regress_api(self):
    """Export to SavedModel format with regress API."""
    columns = ["f1", "i1", "label_regress"]
    model = specialized_learners.RandomForestLearner(
        label="label_regress",
        num_trees=10,
        task=generic_learner.Task.REGRESSION,
    ).train(self.create_dataset_v2(columns))
    test_ds = self.create_dataset_v2(columns[:-1])
    ydf_predictions = model.predict(test_ds)
    tmp_dir = self.create_tempdir().full_path
    model.to_tensorflow_saved_model(
        path=tmp_dir,
        mode="tf",
        feed_example_proto=False,
        servo_api=True,
    )
    tf_model = tf.saved_model.load(tmp_dir)
    tf_model_predict = tf_model.signatures["serving_default"]
    tf_prediction = tf_model_predict(**test_ds)
    self.assertEqual(tf_prediction["outputs"].shape, (100,))
    npt.assert_equal(ydf_predictions, tf_prediction["outputs"])

  def test_to_tensorflow_saved_model_with_example_proto(self):
    """Export to SavedModel format with serialized example inputs."""

    # Create YDF model
    columns = ["f1", "i1", "i2", "c1", "b1", "b2", "cs1", "label_class_binary"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary",
        num_trees=10,
        features=[("cs1", dataspec.Semantic.CATEGORICAL_SET)],
        include_all_columns=True,
    ).train(self.create_dataset_v2(columns))

    test_ds = self.create_dataset_v2(columns[:-1])

    # Save model
    tmp_dir = self.create_tempdir().full_path

    def pre_processing(features):
      features = features.copy()
      features["f1"] = features["f1"] * 2
      return features

    def post_processing(output):
      return output * 3

    # Model predictions with all transformations
    ydf_no_post_process_predictions = model.predict(pre_processing(test_ds))
    # Add extra dimension: ydf squeeze its predictions, while the servo api
    # expects non-squeezed predictions.
    ydf_no_post_process_predictions = np.stack(
        [
            1.0 - ydf_no_post_process_predictions,
            ydf_no_post_process_predictions,
        ],
        axis=1,
    )
    ydf_predictions = post_processing(ydf_no_post_process_predictions)

    model.to_tensorflow_saved_model(
        tmp_dir,
        mode="tf",
        feature_dtypes={"i2": tf.float32, "b2": tf.float32},
        pre_processing=pre_processing,
        post_processing=post_processing,
        servo_api=True,
        feed_example_proto=True,
    )

    # Load model
    tf_model = tf.saved_model.load(tmp_dir)

    # Raw predictions
    tf_test_ds = {
        "f1": tf.constant(test_ds["f1"]),
        "i1": tf.constant(test_ds["i1"]),
        "i2": tf.constant(test_ds["i2"], dtype=tf.float32),
        "c1": tf.constant(test_ds["c1"]),
        "b1": tf.constant(test_ds["b1"]),
        "b2": tf.constant(test_ds["b2"], dtype=tf.float32),
        "cs1": tf.ragged.constant(test_ds["cs1"]),
    }
    raw_tf_predictions = tf_model(tf_test_ds)
    npt.assert_array_equal(ydf_predictions, raw_tf_predictions)

    # Stored pre and post processing.
    for feature in columns[:-2]:
      npt.assert_array_equal(
          tf_model.pre_processing(tf_test_ds)[feature],
          pre_processing(tf_test_ds)[feature],
      )

    npt.assert_array_equal(
        tf_model.post_processing(tf.constant([[1.0, 2.0]], tf.float32)),
        post_processing(tf.constant([[1.0, 2.0]], tf.float32)),
    )

    # Servo API predictions
    tf_model_predict = tf_model.signatures["serving_default"]
    tf_test_ds = []
    for example_idx in range(100):
      tf_test_ds.append(
          tf.train.Example(
              features=tf.train.Features(
                  feature={
                      "f1": tf.train.Feature(
                          float_list=tf.train.FloatList(
                              value=[test_ds["f1"][example_idx]]
                          )
                      ),
                      "i1": tf.train.Feature(
                          int64_list=tf.train.Int64List(
                              value=[test_ds["i1"][example_idx]]
                          )
                      ),
                      "i2": tf.train.Feature(
                          float_list=tf.train.FloatList(
                              value=[test_ds["i2"][example_idx]]
                          )
                      ),
                      "c1": tf.train.Feature(
                          bytes_list=tf.train.BytesList(
                              value=[bytes(test_ds["c1"][example_idx], "utf-8")]
                          )
                      ),
                      "b1": tf.train.Feature(
                          int64_list=tf.train.Int64List(
                              value=[test_ds["b1"][example_idx]]
                          )
                      ),
                      "b2": tf.train.Feature(
                          float_list=tf.train.FloatList(
                              value=[test_ds["b2"][example_idx]]
                          )
                      ),
                      "cs1": tf.train.Feature(
                          bytes_list=tf.train.BytesList(
                              value=[
                                  bytes(x, "utf-8")
                                  for x in test_ds["cs1"][example_idx]
                              ]
                          )
                      ),
                  }
              )
          ).SerializeToString()
      )
    tf_predictions = tf_model_predict(inputs=tf_test_ds)
    npt.assert_equal(ydf_predictions, tf_predictions["scores"])

  @parameterized.parameters(True, False)
  def test_to_tensorflow_saved_model_with_example_proto_multidim(
      self, with_filter: bool
  ):
    """Export to SavedModel format with serialized example inputs."""

    # Create YDF model
    columns = [
        "multi_f1",
        "multi_i1",
        "multi_c1",
        "multi_b1",
        "label_class_binary",
    ]
    feature_columns = columns[:-2] if with_filter else columns[:-1]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary",
        num_trees=10,
        features=feature_columns if with_filter else None,
    ).train(self.create_dataset_v2(columns))

    # Golden predictions
    test_ds = self.create_dataset_v2(feature_columns)

    # Save model
    tmp_dir = self.create_tempdir().full_path

    # Golden predictions
    ydf_predictions = model.predict(test_ds)
    # Add extra dimension: ydf squeeze its predictions, while the servo api
    # expects non-squeezed predictions.
    ydf_predictions = np.stack(
        [
            1.0 - ydf_predictions,
            ydf_predictions,
        ],
        axis=1,
    )

    model.to_tensorflow_saved_model(
        tmp_dir, mode="tf", servo_api=True, feed_example_proto=True
    )

    # Load model
    tf_model = tf.saved_model.load(tmp_dir)

    # Raw predictions
    tf_test_ds = {k: tf.constant(test_ds[k]) for k in feature_columns}
    raw_tf_predictions = tf_model(tf_test_ds)
    npt.assert_array_equal(ydf_predictions, raw_tf_predictions)

    # Servo API predictions
    tf_model_predict = tf_model.signatures["serving_default"]
    tf_test_ds = []
    for example_idx in range(100):
      proto_feature = {
          "multi_f1": tf.train.Feature(
              float_list=tf.train.FloatList(
                  value=test_ds["multi_f1"][example_idx][:]
              )
          ),
          "multi_i1": tf.train.Feature(
              int64_list=tf.train.Int64List(
                  value=test_ds["multi_i1"][example_idx][:]
              )
          ),
          "multi_c1": tf.train.Feature(
              bytes_list=tf.train.BytesList(
                  value=[
                      bytes(x, "utf-8")
                      for x in test_ds["multi_c1"][example_idx]
                  ]
              )
          ),
      }
      if not with_filter:
        proto_feature["multi_b1"] = tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=test_ds["multi_b1"][example_idx][:]
            )
        )
      tf_test_ds.append(
          tf.train.Example(
              features=tf.train.Features(feature=proto_feature)
          ).SerializeToString()
      )
    tf_predictions = tf_model_predict(inputs=tf_test_ds)
    npt.assert_equal(ydf_predictions, tf_predictions["scores"])

  def test_to_tensorflow_saved_model_with_resource_postprocessing(self):
    """Test having the post processing be resource dependent."""

    columns = ["f1", "f2", "label_class_binary"]
    m1_ds = self.create_dataset_v2(columns)
    m1_ydf = specialized_learners.RandomForestLearner(
        label="label_class_binary",
        num_trees=10,
        task=generic_learner.Task.CLASSIFICATION,
    ).train(m1_ds)
    m1_tf = m1_ydf.to_tensorflow_function()

    class PreProcessing(tf.Module):

      def __call__(self, features):
        features = features.copy()
        features["m1"] = m1_tf(features)
        return features

    pre_processing = PreProcessing()
    pre_processing.m1_tf = m1_tf

    m2_ds = {**m1_ds, "m1": m1_ydf.predict(m1_ds)}
    m2 = specialized_learners.RandomForestLearner(
        label="label_class_binary",
        num_trees=10,
        task=generic_learner.Task.CLASSIFICATION,
    ).train(m2_ds)

    tmp_dir = self.create_tempdir().full_path
    m2.to_tensorflow_saved_model(
        path=tmp_dir, mode="tf", pre_processing=pre_processing
    )

    _ = tf.saved_model.load(tmp_dir)

  @parameterized.parameters(
      (False,),
      (True,),
  )
  def test_to_tensorflow_saved_model_with_tf_function_processing(
      self, serialized_input: bool
  ):
    """Use a @tf.function to process model inputs and export the model."""

    # Creates a dataset.
    columns = ["f1", "f2", "label_class_binary"]
    raw_dict_ds = self.create_dataset_v2(columns)

    # Define a preprocessing TensorFlow function for the dataset.
    @tf.function
    def preprocessing(raw_features: Dict[str, tf.Tensor]):
      features = {**raw_features}
      # Create a new feature.
      features["sin_f1"] = tf.sin(features["f1"])
      # Remove a feature
      del features["f1"]
      return features

    # Create a TF dataset containing the raw Numpy dataset and the preprocessing
    # function.
    processed_tf_dataset = (
        tf.data.Dataset.from_tensor_slices(raw_dict_ds)
        .batch(128)  # The batch size has no impact on the model.
        .map(preprocessing)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Train a model on the preprocessed dataset.
    learner = specialized_learners.RandomForestLearner(
        label="label_class_binary",
        num_trees=10,
        task=generic_learner.Task.CLASSIFICATION,
    )
    ydf_model = learner.train(processed_tf_dataset)

    # Generate the golden predictions.
    ydf_predictions = ydf_model.predict(processed_tf_dataset)

    # Check that the model has been trained on the pre-processed features.
    self.assertSequenceEqual(ydf_model.input_feature_names(), ["f2", "sin_f1"])

    model_path = self.create_tempdir().full_path

    if not serialized_input:
      ydf_model.to_tensorflow_saved_model(
          path=model_path,
          mode="tf",
          pre_processing=preprocessing,
          tensor_specs={
              "f1": tf.TensorSpec(shape=[None], name="f1", dtype=tf.float64),
              "f2": tf.TensorSpec(shape=[None], name="f2", dtype=tf.float64),
          },
      )

      # Check that the exported model works.
      tf_model = tf.saved_model.load(model_path)
      tf_dataset = {
          k: tf.constant(v) for k, v in raw_dict_ds.items() if "label" not in k
      }
      tf_predictions = tf_model(tf_dataset)
    else:
      ydf_model.to_tensorflow_saved_model(
          path=model_path,
          mode="tf",
          pre_processing=preprocessing,
          feed_example_proto=True,
          feature_specs={
              "f1": tf.io.FixedLenFeature(
                  shape=[], dtype=tf.float32, default_value=math.nan
              ),
              "f2": tf.io.FixedLenFeature(
                  shape=[], dtype=tf.float32, default_value=math.nan
              ),
          },
      )

      tf_model = tf.saved_model.load(model_path)
      tf_dataset = [
          tf.train.Example(
              features=tf.train.Features(
                  feature={
                      "f1": tf.train.Feature(
                          float_list=tf.train.FloatList(
                              value=[raw_dict_ds["f1"][i]]
                          )
                      ),
                      "f2": tf.train.Feature(
                          float_list=tf.train.FloatList(
                              value=[raw_dict_ds["f2"][i]]
                          )
                      ),
                  }
              )
          ).SerializeToString()
          for i in range(100)
      ]
      tf_predictions = tf_model.signatures["serving_default"](
          inputs=tf_dataset
      )["output"]

    npt.assert_equal(ydf_predictions, tf_predictions)

  def test_to_tensorflow_saved_model_with_processing_and_wrong_spec(self):

    columns = ["f1", "label_class_binary"]
    raw_dict_dataset = self.create_dataset_v2(columns)

    # Define a preprocessing tensorflow function for the dataset.
    @tf.function
    def preprocessing(raw_features: Dict[str, tf.Tensor]):
      features = {**raw_features}
      # Create a new feature.
      features["sin_f1"] = tf.sin(features["f1"])
      # Remove a feature.
      del features["f1"]
      return features

    processed_tf_dataset = (
        tf.data.Dataset.from_tensor_slices(raw_dict_dataset)
        .batch(128)
        .map(preprocessing)
        .prefetch(tf.data.AUTOTUNE)
    )

    learner = specialized_learners.RandomForestLearner(
        label="label_class_binary",
        num_trees=10,
        task=generic_learner.Task.CLASSIFICATION,
    )
    ydf_model = learner.train(processed_tf_dataset)

    model_path = self.create_tempdir().full_path

    with self.assertRaisesRegex(
        ValueError,
        "This error might be caused by the feature spec",
    ):
      ydf_model.to_tensorflow_saved_model(
          path=model_path,
          mode="tf",
          pre_processing=preprocessing,
          feed_example_proto=True,
      )

  def test_to_tensorflow_saved_model_adult_classify_api_serialized_examples(
      self,
  ):
    """Export to SavedModel format of a model trained from file."""

    train_ds_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_train.recordio"
    )
    test_ds_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.recordio"
    )

    model = specialized_learners.RandomForestLearner(
        label="income",
        num_trees=10,
    ).train(f"tfrecordv2+tfe:{train_ds_path}")

    ydf_predictions = model.predict(f"tfrecordv2+tfe:{test_ds_path}")

    tempdir = self.create_tempdir().full_path

    # TODO: Implement automatic dtype.
    model.to_tensorflow_saved_model(
        tempdir,
        mode="tf",
        servo_api=True,
        feed_example_proto=True,
    )

    tf_model = tf.saved_model.load(tempdir)
    tf_predict = tf_model.signatures["serving_default"]

    for example_idx, serialized_example in enumerate(
        tf.data.TFRecordDataset([test_ds_path]).take(10)
    ):
      tf_prediction = tf_predict(inputs=[serialized_example])
      npt.assert_array_equal(
          tf_prediction["scores"][:, 1], ydf_predictions[example_idx]
      )
      npt.assert_array_equal(tf_prediction["classes"], [[b"<=50K", b">50K"]])

  def test_to_tensorflow_saved_model_wrong_dtype(self):
    columns = ["f1", "label_class_binary"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary", num_trees=10
    ).train(self.create_dataset_v2(columns))
    tmp_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(
        ValueError,
        "expected to have type \\[tf.float16, tf.float32, tf.float64\\] or"
        " \\[tf.int8, tf.int16, tf.int32, tf.int64, tf.uint8, tf.uint16,"
        " tf.uint32, tf.uint64\\]",
    ):
      model.to_tensorflow_saved_model(
          tmp_dir,
          mode="tf",
          feed_example_proto=False,
          feature_dtypes={"f1": tf.string},
      )

  def test_tensorflow_raw_input_signature_default(self):
    columns = ["f1", "i1", "c1", "b1", "label_class_binary"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary", num_trees=10
    ).train(self.create_dataset_v2(columns))
    self.assertEqual(
        export_tf.tensorflow_raw_input_signature(model, {}),
        {
            "f1": tf.TensorSpec(shape=(None,), dtype=tf.float64, name="f1"),
            "i1": tf.TensorSpec(shape=(None,), dtype=tf.int64, name="i1"),
            "c1": tf.TensorSpec(shape=(None,), dtype=tf.string, name="c1"),
            "b1": tf.TensorSpec(shape=(None,), dtype=tf.bool, name="b1"),
        },
    )

  def test_tensorflow_raw_input_signature_multidim(self):
    columns = [
        "multi_f1",
        "multi_i1",
        "multi_c1",
        "multi_b1",
        "label_class_binary",
    ]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary", num_trees=10
    ).train(self.create_dataset_v2(columns))
    self.assertEqual(
        export_tf.tensorflow_raw_input_signature(model, {}),
        {
            "multi_f1": tf.TensorSpec(
                shape=(None, 5), dtype=tf.float64, name="multi_f1"
            ),
            "multi_i1": tf.TensorSpec(
                shape=(None, 5), dtype=tf.int64, name="multi_i1"
            ),
            "multi_c1": tf.TensorSpec(
                shape=(None, 5), dtype=tf.string, name="multi_c1"
            ),
            "multi_b1": tf.TensorSpec(
                shape=(None, 5), dtype=tf.bool, name="multi_b1"
            ),
        },
    )

  def test_tensorflow_raw_input_signature_override(self):
    columns = ["f1", "i1", "c1", "b1", "label_class_binary"]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary", num_trees=10
    ).train(self.create_dataset_v2(columns))
    self.assertEqual(
        export_tf.tensorflow_raw_input_signature(
            model,
            {"f1": tf.int32, "i1": tf.int64, "c1": tf.string, "b1": tf.float16},
        ),
        {
            "f1": tf.TensorSpec(shape=(None,), dtype=tf.int32, name="f1"),
            "i1": tf.TensorSpec(shape=(None,), dtype=tf.int64, name="i1"),
            "c1": tf.TensorSpec(shape=(None,), dtype=tf.string, name="c1"),
            "b1": tf.TensorSpec(shape=(None,), dtype=tf.float16, name="b1"),
        },
    )

  def test_tensorflow_raw_input_multidim_signature_override(self):
    columns = [
        "multi_f1",
        "multi_i1",
        "multi_c1",
        "multi_b1",
        "label_class_binary",
    ]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary", num_trees=10
    ).train(self.create_dataset_v2(columns))
    self.assertEqual(
        export_tf.tensorflow_raw_input_signature(
            model,
            {
                "multi_f1": tf.int32,
                "multi_i1": tf.int64,
                "multi_c1": tf.string,
                "multi_b1": tf.float16,
            },
        ),
        {
            "multi_f1": tf.TensorSpec(
                shape=(None, 5), dtype=tf.int32, name="multi_f1"
            ),
            "multi_i1": tf.TensorSpec(
                shape=(None, 5), dtype=tf.int64, name="multi_i1"
            ),
            "multi_c1": tf.TensorSpec(
                shape=(None, 5), dtype=tf.string, name="multi_c1"
            ),
            "multi_b1": tf.TensorSpec(
                shape=(None, 5), dtype=tf.float16, name="multi_b1"
            ),
        },
    )

  def test_tensorflow_feature_spec_default(self):
    columns = [
        "f1",
        "i1",
        "c1",
        "b1",
        "multi_f1",
        "multi_i1",
        "multi_c1",
        "multi_b1",
        "label_class_binary",
    ]
    model = specialized_learners.RandomForestLearner(
        label="label_class_binary", num_trees=10
    ).train(self.create_dataset_v2(columns))
    self.assertEqual(
        export_tf.tensorflow_feature_spec(model, {}),
        {
            "f1": tf.io.FixedLenFeature(
                shape=[], dtype=tf.float32, default_value=math.nan
            ),
            "i1": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "c1": tf.io.FixedLenFeature(
                shape=[], dtype=tf.string, default_value=""
            ),
            "b1": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "multi_f1": tf.io.FixedLenFeature(
                shape=[5], dtype=tf.float32, default_value=[math.nan] * 5
            ),
            "multi_i1": tf.io.FixedLenFeature(shape=[5], dtype=tf.int64),
            "multi_c1": tf.io.FixedLenFeature(
                shape=[5],
                dtype=tf.string,
                default_value=[""] * 5,
            ),
            "multi_b1": tf.io.FixedLenFeature(shape=[5], dtype=tf.int64),
        },
    )

  def test_usage_example_to_tensorflow_function(self):
    """Usage example of the "to_tensorflow_function" method."""

    # Train a model.
    model = specialized_learners.RandomForestLearner(label="l").train({
        "f1": np.random.random(size=100),
        "f2": np.random.random(size=100),
        "l": np.random.randint(2, size=100),
    })

    # Convert model to a TF module.
    tf_model = model.to_tensorflow_function()

    # Make predictions with the TF module.
    tf_predictions = tf_model({
        "f1": tf.constant([0, 0.5, 1]),
        "f2": tf.constant([1, 0, 0.5]),
    })

  def test_usage_example_to_tensorflow_saved_model(self):
    """Usage example of the "to_tensorflow_saved_model" method."""
    ydf = specialized_learners

    # Part 1

    # Train a model.
    model = ydf.RandomForestLearner(label="l").train({
        "f1": np.random.random(size=100),
        "f2": np.random.random(size=100).astype(dtype=np.float32),
        "l": np.random.randint(2, size=100),
    })

    # Export the model to the TensorFlow SavedModel format.
    model.to_tensorflow_saved_model(path="/tmp/my_model", mode="tf")

    # Load the saved model.
    tf_model = tf.saved_model.load("/tmp/my_model")

    # Make predictions
    tf_predictions = tf_model({
        "f1": tf.constant(np.random.random(size=10)),
        "f2": tf.constant(np.random.random(size=10), dtype=tf.float32),
    })

    # Part 3
    model.to_tensorflow_saved_model(
        path="/tmp/my_model",
        mode="tf",
        # "f1" is fed as an tf.int64 instead of tf.float64
        feature_dtypes={"f1": tf.int64},
    )

    # Part 4
    def pre_processing(features):
      features = features.copy()
      features["f1"] = features["f1"] * 2
      return features

    model.to_tensorflow_saved_model(
        path="/tmp/my_model",
        mode="tf",
        pre_processing=pre_processing,
    )

  @parameterized.product(mode=["tf", "keras"])
  def test_export_non_unicode_feature_values(self, mode):
    text = "feature,label\nCafé,oné\nfoobar,zéro"
    encoded_text = text.encode("windows-1252")
    with self.assertRaises(UnicodeDecodeError):
      encoded_text.decode()
    data_path = self.create_tempfile().full_path
    model_path = self.create_tempdir().full_path
    with open(data_path, "wb") as f:
      f.write(encoded_text)
    model = specialized_learners.CartLearner(
        label="label",
        min_examples=1,
        min_vocab_frequency=1,
    ).train("csv:" + data_path)
    ydf_predictions = model.predict("csv:" + data_path)

    model.to_tensorflow_saved_model(path=model_path, mode=mode)

    if mode == "keras":
      tf_dataset = tf.data.Dataset.from_tensor_slices({
          "feature": np.array(
              ["Café".encode("windows-1252"), "foobar".encode("windows-1252")]
          ).reshape(-1, 1)
      })
      tfdf_model = tf.keras.models.load_model(model_path)
      tfdf_predictions = tfdf_model.predict(tf_dataset)
      npt.assert_equal(tfdf_predictions.flatten(), ydf_predictions)


if __name__ == "__main__":
  absltest.main()
