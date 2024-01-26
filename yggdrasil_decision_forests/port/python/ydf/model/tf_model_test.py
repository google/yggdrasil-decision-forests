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

import os
from typing import Callable, List, Mapping, Optional, Tuple

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

from ydf.learner import specialized_learners
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


if __name__ == "__main__":
  absltest.main()
