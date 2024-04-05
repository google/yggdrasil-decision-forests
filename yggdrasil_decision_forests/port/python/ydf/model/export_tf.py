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

"""Utilities to export TF models."""

import math
import shutil
import tempfile
from typing import Any, Callable, Dict, Literal, Optional
import uuid

from ydf.dataset import dataspec
from ydf.model import generic_model
from ydf.utils import log


_ERROR_MESSAGE_MISSING_TF = (
    '"tensorflow" is needed by this function. Make sure it'
    " installed and try again. If using pip, run `pip install"
    " tensorflow`. If using Bazel/Blaze, add a dependency to"
    " TensorFlow."
)

_ERROR_MESSAGE_MISSING_TFDF = (
    '"tensorflow_decision_forests" is needed by this function. Make sure it'
    " installed and try again. If using pip, run `pip install"
    " tensorflow_decision_forests`. If using Bazel/Blaze, add a dependency to"
    " TensorFlow Decision Forests."
)

TFDType = Any  # TensorFlow DType e.g. tf.float32


def ydf_model_to_tensorflow_saved_model(
    ydf_model: "generic_model.GenericModel",
    path: str,
    input_model_signature_fn: Any,
    mode: Literal["keras", "tf"],
    feature_dtypes: Dict[str, TFDType],
    servo_api: bool,
    feed_example_proto: bool,
    pre_processing: Optional[Callable],  # pylint: disable=g-bare-generic
    post_processing: Optional[Callable],  # pylint: disable=g-bare-generic
    temp_dir: Optional[str],
):  # pylint: disable=g-doc-args
  """Exports the model as a TensorFlow Saved model.

  See GenericModel.to_tensorflow_saved_model for the documentation.
  """
  if mode == "keras":
    for value, name, expected in [
        (feature_dtypes, "feature_dtypes", {}),
        (servo_api, "servo_api", False),
        (feed_example_proto, "feed_example_proto", False),
        (pre_processing, "pre_processing", None),
        (post_processing, "post_processing", None),
    ]:
      if value != expected:
        raise ValueError(f"{name!r} is not supported for `keras` mode.")
    ydf_model_to_tensorflow_saved_model_keras_mode(
        ydf_model=ydf_model,
        path=path,
        input_model_signature_fn=input_model_signature_fn,
        temp_dir=temp_dir,
    )

  elif mode == "tf":
    if input_model_signature_fn is not None:
      raise ValueError(
          "input_model_signature_fn is not supported for `tf` mode."
      )
    ydf_model_to_tensorflow_saved_model_tf_mode(
        ydf_model=ydf_model,
        path=path,
        feature_dtypes=feature_dtypes,
        servo_api=servo_api,
        feed_example_proto=feed_example_proto,
        pre_processing=pre_processing,
        post_processing=post_processing,
        temp_dir=temp_dir,
    )
  else:
    raise ValueError(f"Invalid mode: {mode}")


def ydf_model_to_tensorflow_saved_model_keras_mode(
    ydf_model: "generic_model.GenericModel",
    path: str,
    input_model_signature_fn: Any,
    temp_dir: Optional[str],
):  # pylint: disable=g-doc-args
  tfdf = import_tensorflow_decision_forests()

  # Do not pass input_model_signature_fn if it is None.
  not_none_params = {}
  if input_model_signature_fn is not None:
    not_none_params["input_model_signature_fn"] = input_model_signature_fn
  with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
    ydf_model.save(tmpdirname)
    tfdf.keras.yggdrasil_model_to_keras_model(
        src_path=tmpdirname,
        dst_path=path,
        verbose=log.current_log_level(),
        **not_none_params,
    )


def ydf_model_to_tensorflow_saved_model_tf_mode(
    ydf_model: "generic_model.GenericModel",
    path: str,
    feature_dtypes: Dict[str, TFDType],
    servo_api: bool,
    feed_example_proto: bool,
    pre_processing: Optional[Callable],  # pylint: disable=g-bare-generic
    post_processing: Optional[Callable],  # pylint: disable=g-bare-generic
    temp_dir: Optional[str],
):  # pylint: disable=g-doc-args

  tf = import_tensorflow()

  # The temporary files should remain available until the call to
  # "tf.saved_model.save"
  with tempfile.TemporaryDirectory(dir=temp_dir) as effective_temp_dir:

    tf_module = ydf_model.to_tensorflow_function(
        temp_dir=effective_temp_dir,
        squeeze_binary_classification=not servo_api,
    )

    # Store pre / post processing operations
    # Note: Storing the raw variable allows for pre/post-processing to be
    # TensorFlow modules with resources.
    tf_module.raw_pre_processing = pre_processing
    tf_module.raw_post_processing = post_processing
    tf_module.pre_processing = tf.function(pre_processing or (lambda x: x))
    tf_module.post_processing = tf.function(post_processing or (lambda x: x))

    # Apply pre / post processing to call.
    def call(features):
      return tf_module.post_processing(
          tf_module.raw_call(tf_module.pre_processing(features))
      )

    tf_module.raw_call = tf_module.call
    tf_module.call = call

    # Trace the call function.
    # Note: "tf.saved_model.save" can only export functions that have been
    # traced.
    raw_input_signature = tensorflow_raw_input_signature(
        ydf_model, feature_dtypes
    )
    tf_module.__call__.get_concrete_function(raw_input_signature)

    # Output format
    if servo_api:

      if ydf_model.task() == generic_model.Task.CLASSIFICATION:
        label_classes = ydf_model.label_classes()

        # "classify" Servo API
        def predict_output_format(features, batch_size):
          raw_output = tf_module(features)
          batched_label_classes = tf.broadcast_to(
              input=tf.constant(label_classes),
              shape=(batch_size, len(label_classes)),
          )
          return {"classes": batched_label_classes, "scores": raw_output}

      elif ydf_model.task() == generic_model.Task.REGRESSION:

        # "regress" Servo API
        def predict_output_format(features, batch_size):
          del batch_size
          raw_output = tf_module(features)
          return {"outputs": raw_output}

      else:
        raise ValueError(
            f"servo_api=True non supported for task {ydf_model.task()!r}"
        )

    else:

      # "predict" Servo API i.e. raw output
      # Note: "signature" outputs need to be dictionaries.
      def predict_output_format(features, batch_size):
        del batch_size
        return {"output": tf_module(features)}

    # Input feature formats
    if not feed_example_proto:

      # Feed raw feature values.

      @tf.function
      def predict_input_format(features):
        any_feature = next(iter(features.values()))
        batch_size = tf.shape(any_feature)[0]
        return predict_output_format(features, batch_size)

      signatures = {
          "serving_default": predict_input_format.get_concrete_function(
              raw_input_signature
          )
      }

    else:
      # Feed binary serialized TensorFlow Example protos.
      feature_spec = tensorflow_feature_spec(ydf_model, feature_dtypes)

      @tf.function(
          input_signature=[
              tf.TensorSpec([None], dtype=tf.string, name="inputs")
          ]
      )
      def predict_input_format(
          serialized_examples: tf.Tensor,
      ):
        batch_size = tf.shape(serialized_examples)[0]
        features = tf.io.parse_example(serialized_examples, feature_spec)
        return predict_output_format(features, batch_size)

      signatures = {"serving_default": predict_input_format}

    tf.saved_model.save(tf_module, path, signatures=signatures)


def ydf_model_to_tf_function(  # pytype: disable=name-error
    ydf_model: "generic_model.GenericModel",
    temp_dir: Optional[str],
    can_be_saved: bool,
    squeeze_binary_classification: bool,
) -> "tensorflow.Module":  # pylint: disable=g-doc-args
  """Converts a YDF model to a TensorFlow function.

  See GenericModel.to_tensorflow_function for the documentation.
  """

  tf = import_tensorflow()
  tfdf = import_tensorflow_decision_forests()
  tf_op = tfdf.keras.core.tf_op

  # Using prefixes ensure multiple models can be combined in a single
  # SavedModel.
  file_prefix = uuid.uuid4().hex[:8] + "_"

  # Save the model to disk and load it as a TensorFlow resource.
  tmp_dir = tempfile.mkdtemp(dir=temp_dir)
  try:
    ydf_model.save(
        tmp_dir,
        advanced_options=generic_model.ModelIOOptions(file_prefix=file_prefix),
    )
    op_model = tf_op.ModelV2(tmp_dir, verbose=False, file_prefix=file_prefix)
  finally:

    if not can_be_saved:
      shutil.rmtree(tmp_dir)

  # If "extract_dim" is not None, the model returns the "extract_dim
  # dimension of the output of the TF-DF Predict Op.
  if ydf_model.task() == generic_model.Task.CLASSIFICATION:
    if squeeze_binary_classification and len(ydf_model.label_classes()) == 2:
      extract_dim = 1
    else:
      extract_dim = None
  else:
    # Single dimention outputs (e.g. regression, ranking) is always squeezed.
    extract_dim = 0

  # Wrap the model into a tf module.
  class CallableModule(tf.Module):

    @tf.function
    def __call__(self, features):
      return self.call(features)

  callable_module = CallableModule()

  @tf.function
  def call(features):
    dense_predictions = op_model.apply(features).dense_predictions
    assert len(dense_predictions.shape) == 2
    if extract_dim is not None:
      return dense_predictions[:, extract_dim]
    else:
      return dense_predictions

  callable_module.call = call
  callable_module.op_model = op_model  # Link model resources
  return callable_module


def import_tensorflow():
  """Imports the tensorflow module."""
  try:
    import tensorflow  # pylint: disable=g-import-not-at-top,import-outside-toplevel # pytype: disable=import-error

    return tensorflow
  except ImportError as exc:
    raise ValueError(_ERROR_MESSAGE_MISSING_TF) from exc


def import_tensorflow_decision_forests():
  """Imports the tensorflow decision forests module."""
  try:
    import tensorflow_decision_forests as tfdf  # pylint: disable=g-import-not-at-top,import-outside-toplevel # pytype: disable=import-error

    return tfdf
  except ImportError as exc:
    raise ValueError(_ERROR_MESSAGE_MISSING_TFDF) from exc


def tensorflow_raw_input_signature(
    model: "generic_model.GenericModel",
    feature_dtypes: Dict[str, TFDType],
) -> Dict[str, Any]:
  """A TF input_signature to feed raw feature values into the model."""
  tf = import_tensorflow()

  used_feature_dtypes = set()

  input_signature = {}
  for src_feature in model.input_features():
    user_dtype = feature_dtypes.get(src_feature.name)
    if user_dtype is not None:
      used_feature_dtypes.add(src_feature.name)

    if src_feature.semantic == dataspec.Semantic.NUMERICAL:
      dst_feature = tf.TensorSpec(
          shape=[None], dtype=user_dtype or tf.float32, name=src_feature.name
      )
    elif src_feature.semantic == dataspec.Semantic.CATEGORICAL:
      dst_feature = tf.TensorSpec(
          shape=[None], dtype=user_dtype or tf.string, name=src_feature.name
      )
    elif src_feature.semantic == dataspec.Semantic.BOOLEAN:
      dst_feature = tf.TensorSpec(
          shape=[None], dtype=user_dtype or tf.float32, name=src_feature.name
      )
    elif src_feature.semantic == dataspec.Semantic.CATEGORICAL_SET:
      dst_feature = tf.RaggedTensorSpec(
          shape=[None, None],
          dtype=user_dtype or tf.string,
          name=src_feature.name,
      )
    else:
      raise ValueError(f"Unsupported semantic: {src_feature.semantic}")
    input_signature[src_feature.name] = dst_feature

  diff = set(feature_dtypes).difference(used_feature_dtypes)
  if diff:
    raise ValueError(
        f"Input dtypes for features not used by the model: {list(diff)!r}"
    )

  return input_signature


def tensorflow_feature_spec(
    model: "generic_model.GenericModel",
    feature_dtypes: Dict[str, TFDType],
) -> Dict[str, Any]:
  """A TF feature spec used to deserialize tf example protos."""
  tf = import_tensorflow()

  def missing_value(tf_dtype):
    """Representation of a missing value; if possible."""
    # Follow the missing value representation described in ydf.Semantic.
    if tf_dtype.is_floating:
      return math.nan
    elif tf_dtype == tf.string:
      return ""
    else:
      return None  # Missing value not allowed

  feature_spec = {}
  for src_feature in model.input_features():
    user_dtype = feature_dtypes.get(src_feature.name)

    if src_feature.semantic == dataspec.Semantic.NUMERICAL:
      effective_dtype = user_dtype or tf.float32
      dst_feature = tf.io.FixedLenFeature(
          shape=[],
          dtype=effective_dtype,
          default_value=missing_value(effective_dtype),
      )
    elif src_feature.semantic == dataspec.Semantic.CATEGORICAL:
      effective_dtype = user_dtype or tf.string
      dst_feature = tf.io.FixedLenFeature(
          shape=[],
          dtype=effective_dtype,
          default_value=missing_value(effective_dtype),
      )
    elif src_feature.semantic == dataspec.Semantic.BOOLEAN:
      effective_dtype = user_dtype or tf.float32
      dst_feature = tf.io.FixedLenFeature(
          shape=[],
          dtype=effective_dtype,
          default_value=missing_value(effective_dtype),
      )
    elif src_feature.semantic == dataspec.Semantic.CATEGORICAL_SET:
      dst_feature = tf.io.VarLenFeature(dtype=user_dtype or tf.string)
    else:
      raise ValueError(f"Unsupported semantic: {src_feature.semantic}")
    feature_spec[src_feature.name] = dst_feature

  return feature_spec
