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

import logging
import math
import shutil
import tempfile
from typing import Any, Callable, Dict, Literal, Optional, Sequence
import uuid

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.dataset import dataspec
from ydf.dataset.io import dataset_io
from ydf.model import generic_model
from ydf.utils import log

# Bypass the dependency checker + returns details explanations about why
# importing tf or tf-df fails (e.g. missing dependency, missing or invalid .so
# file).
# pytype: disable=import-error
# pylint: disable=g-import-not-at-top
try:
  import tensorflow as tf
  import tensorflow_decision_forests as tfdf
except ImportError as exc:
  raise ImportError(
      "Cannot import tensorflow or tensorflow_decision_forests."
  ) from exc
# pylint: enable=g-import-not-at-top
# pytype: enable=import-error


TFDType = Any  # TensorFlow DType e.g. tf.float32
TFTensor = Any  # A TensorFlow Tensor i.e. tensorflow.Tensor

# Mapping between YDF dtype and TF dtypes.
_YDF_DTYPE_TO_TF_DTYPE: Dict["ds_pb.DType", TFDType] = None

# Mapping TF dtypes to the TF dtype compatible with tensorflow example.
# Note that tensorflow example proto only support tf.int64, tf.float32,
# and tf.string dtypes.
_TF_DTYPE_TO_TF_EXAMPLE_DTYPE: Dict[TFDType, TFDType] = None


def mapping_ydf_dtype_to_tf_dtype() -> Dict["ds_pb.DType", TFDType]:
  """Mapping between YDF dtype and TF dtypes."""

  global _YDF_DTYPE_TO_TF_DTYPE
  if _YDF_DTYPE_TO_TF_DTYPE is None:
    _YDF_DTYPE_TO_TF_DTYPE = {
        ds_pb.DType.DTYPE_INT8: tf.int8,
        ds_pb.DType.DTYPE_INT16: tf.int16,
        ds_pb.DType.DTYPE_INT32: tf.int32,
        ds_pb.DType.DTYPE_INT64: tf.int64,
        ds_pb.DType.DTYPE_UINT8: tf.uint8,
        ds_pb.DType.DTYPE_UINT16: tf.uint16,
        ds_pb.DType.DTYPE_UINT32: tf.uint32,
        ds_pb.DType.DTYPE_UINT64: tf.uint64,
        ds_pb.DType.DTYPE_FLOAT16: tf.float16,
        ds_pb.DType.DTYPE_FLOAT32: tf.float32,
        ds_pb.DType.DTYPE_FLOAT64: tf.float64,
        ds_pb.DType.DTYPE_BOOL: tf.bool,
        ds_pb.DType.DTYPE_BYTES: tf.string,
    }
  return _YDF_DTYPE_TO_TF_DTYPE


def mapping_tf_dtype_to_tf_example_dtype() -> Dict[TFDType, TFDType]:
  """Mapping TF dtypes to the TF dtype compatible with tensorflow example."""

  global _TF_DTYPE_TO_TF_EXAMPLE_DTYPE
  if _TF_DTYPE_TO_TF_EXAMPLE_DTYPE is None:
    _TF_DTYPE_TO_TF_EXAMPLE_DTYPE = {
        tf.int8: tf.int64,
        tf.int16: tf.int64,
        tf.int32: tf.int64,
        tf.int64: tf.int64,
        tf.uint8: tf.int64,
        tf.uint16: tf.int64,
        tf.uint32: tf.int64,
        tf.uint64: tf.int64,
        tf.float16: tf.float32,
        tf.float32: tf.float32,
        tf.float64: tf.float32,
        tf.bool: tf.int64,
        tf.string: tf.string,
    }
  return _TF_DTYPE_TO_TF_EXAMPLE_DTYPE


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
    tensor_specs: Optional[Dict[str, Any]] = None,
    feature_specs: Optional[Dict[str, Any]] = None,
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
        (tensor_specs, "tensor_specs", None),
        (feature_specs, "feature_specs", None),
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
        tensor_specs=tensor_specs,
        feature_specs=feature_specs,
    )
  else:
    raise ValueError(f"Invalid mode: {mode}")


def ydf_model_to_tensorflow_saved_model_keras_mode(
    ydf_model: "generic_model.GenericModel",
    path: str,
    input_model_signature_fn: Any,
    temp_dir: Optional[str],
):  # pylint: disable=g-doc-args

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
    tensor_specs: Optional[Dict[str, Any]] = None,
    feature_specs: Optional[Dict[str, Any]] = None,
):  # pylint: disable=g-doc-args

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
    # traced. However, the function will be traced automatically when
    # "feed_example_proto=True" (i.e., it is okay to skip this block).
    raw_input_signature = None
    if not (
        pre_processing is not None
        and feed_example_proto
        and tensor_specs is None
        and not feature_dtypes
    ):
      if tensor_specs is not None:
        # Use user provided signature.
        raw_input_signature = tensor_specs
      else:
        # Get input signature from the YDF model.
        raw_input_signature = tensorflow_raw_input_signature(
            ydf_model, feature_dtypes
        )

      try:
        tf_module.__call__.get_concrete_function(raw_input_signature)
      # Note: We are really looking for KeyError, but TF wrapps them into
      # interal staging errors.
      except Exception as e:
        if (
            pre_processing is not None
            and tensor_specs is None
            and feature_dtypes is None
        ):
          raise ValueError(
              "Since `tensor_specs` and `feature_dtypes` are not specified, the"
              " tensor specs of `pre_processing` uses the tensor spec of the"
              " model. If `pre_processing` consumes features that are different"
              " (e.g. new features, removed features, change the dtype of the"
              " features) than the features consumed by the model, you need to"
              " set `tensor_specs` or `feature_dtypes` with the tensor spec of"
              " `pre_processing`.\n\nFor example, if `pre_processing` expects a"
              " float32 numerical feature called `f1`, set `tensor_specs ="
              ' {"f1": tf.TensorSpec(shape=[None], name="f1",'
              " dtype=tf.float64)}`."
          ) from e
        raise e

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
      assert raw_input_signature is not None

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
      if feature_specs is None:
        # If not provided by the user, generate the feature specs using the
        # model.
        feature_specs = tensorflow_feature_spec(ydf_model, feature_dtypes)

      @tf.function(
          input_signature=[
              tf.TensorSpec([None], dtype=tf.string, name="inputs")
          ]
      )
      def predict_input_format(
          serialized_examples: tf.Tensor,
      ):
        batch_size = tf.shape(serialized_examples)[0]
        features = tf.io.parse_example(serialized_examples, feature_specs)
        return predict_output_format(features, batch_size)

      signatures = {"serving_default": predict_input_format}

    try:
      tf.saved_model.save(tf_module, path, signatures=signatures)
      # Note: We are really looking for KeyError, but TF wrapps them into
      # interal staging errors.
    except Exception as e:
      if pre_processing is not None and feed_example_proto:
        raise ValueError(
            "This error might be caused by the feature spec (i.e.,"
            " the parsing feature specs of `tf.io.parse_example`) of"
            " the `pre_processing` tf.function argument beeing"
            " different from the specs of the model. If"
            " `pre_processing` consumes features that are different (e.g. new"
            " features, removed features, change the dtype of the features)"
            " than the features consumed by the model, set the"
            " `feature_specs` argument with the feature spec expected by"
            " `pre_processing` and ensure that the output of `pre_processing`"
            " is compatible with the model.\n\nFor example, if"
            " `pre_processing` expects a float32 numerical feature called"
            ' `f1`, set `feature_specs = {"f1": tf.io.FixedLenFeature('
            " shape=[], dtype=tf.float32, default_value=math.nan)}`."
        ) from e
      raise e


def ydf_model_to_tf_function(  # pytype: disable=name-error
    ydf_model: "generic_model.GenericModel",
    temp_dir: Optional[str],
    can_be_saved: bool,
    squeeze_binary_classification: bool,
) -> "tensorflow.Module":  # pylint: disable=g-doc-args
  """Converts a YDF model to a TensorFlow function.

  See GenericModel.to_tensorflow_function for the documentation.
  """

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
    # Single dimension outputs (e.g. regression, ranking) is always squeezed.
    extract_dim = 0

  model_dataspec = ydf_model.data_spec()
  input_features = ydf_model.input_feature_names()

  # Wrap the model into a tf module.
  class CallableModule(tf.Module):

    @tf.function
    def __call__(self, features):
      return self.call(features)

  callable_module = CallableModule()

  @tf.function
  def call(features):
    unrolled_features = _unroll_dict(features, model_dataspec, input_features)
    dense_predictions = op_model.apply(unrolled_features).dense_predictions
    assert len(dense_predictions.shape) == 2
    if extract_dim is not None:
      return dense_predictions[:, extract_dim]
    else:
      return dense_predictions

  callable_module.call = call
  callable_module.op_model = op_model  # Link model resources
  return callable_module


def tf_feature_dtype(
    feature: "generic_model.InputFeature",
    model_dataspec: ds_pb.DataSpecification,
    user_dtypes: Dict[str, TFDType],
) -> TFDType:
  """Determines the TF Dtype of a feature."""

  return tf_feature_dtype_manual(
      feature.name,
      feature.column_idx,
      model_dataspec,
      user_dtypes,
  )


def tf_feature_dtype_manual(
    feature_name: str,
    column_idx: int,
    model_dataspec: ds_pb.DataSpecification,
    user_dtypes: Dict[str, TFDType],
) -> TFDType:
  """Determines the TF Dtype of a feature."""

  # User specified dtype.
  user_dtype = user_dtypes.get(feature_name)
  if user_dtype is not None:
    return user_dtype

  # DType from training dataset
  column_spec = model_dataspec.columns[column_idx]
  if column_spec.HasField("dtype"):
    tf_dtype = mapping_ydf_dtype_to_tf_dtype().get(column_spec.dtype)
    if tf_dtype is None:
      raise ValueError(f"Unsupported dtype: {column_spec.dtype}")
    return tf_dtype

  # DType from feature semantic
  if column_spec.type == ds_pb.NUMERICAL:
    return tf.float32
  elif column_spec.type == ds_pb.CATEGORICAL:
    return tf.string
  elif column_spec.type == ds_pb.BOOLEAN:
    return tf.int64
  elif column_spec.type == ds_pb.CATEGORICAL_SET:
    return tf.string
  else:
    raise ValueError(f"Unsupported semantic: {column_spec.type}")


def tensorflow_raw_input_signature(
    model: "generic_model.GenericModel",
    feature_dtypes: Dict[str, TFDType],
) -> Dict[str, Any]:
  """A TF input_signature to feed raw feature values into the model."""

  model_dataspec = model.data_spec()
  input_features = model.input_features()
  input_feature_names_set = set(f.name for f in input_features)

  input_signature = {}

  # Multi-dim features
  for unstacked in model_dataspec.unstackeds:
    if unstacked.size == 0:
      raise RuntimeError("Empty unstacked")
    sub_names = dataset_io.unrolled_feature_names(
        unstacked.original_name, unstacked.size
    )
    # Note: The "input_features" contain unrolled feature names.
    if sub_names[0] not in input_feature_names_set:
      continue

    tf_dtype = tf_feature_dtype_manual(
        unstacked.original_name,
        unstacked.begin_column_idx,
        model_dataspec,
        feature_dtypes,
    )

    if unstacked.type in [
        ds_pb.ColumnType.NUMERICAL,
        ds_pb.ColumnType.DISCRETIZED_NUMERICAL,
        ds_pb.ColumnType.CATEGORICAL,
        ds_pb.ColumnType.BOOLEAN,
    ]:
      dst_feature = tf.TensorSpec(
          shape=[None, unstacked.size],
          dtype=tf_dtype,
          name=unstacked.original_name,
      )
    else:
      raise ValueError(
          f"Unsupported semantic {unstacked.type} for multi-dim feature"
          f" {unstacked.original_name!r}"
      )
    input_signature[unstacked.original_name] = dst_feature

  # Single-dim features
  for src_feature in input_features:
    column = model_dataspec.columns[src_feature.column_idx]
    if column.is_unstacked:
      continue

    tf_dtype = tf_feature_dtype(src_feature, model_dataspec, feature_dtypes)

    if column.type in [
        ds_pb.ColumnType.NUMERICAL,
        ds_pb.ColumnType.DISCRETIZED_NUMERICAL,
        ds_pb.ColumnType.CATEGORICAL,
        ds_pb.ColumnType.BOOLEAN,
    ]:
      dst_feature = tf.TensorSpec(
          shape=[None], dtype=tf_dtype, name=src_feature.name
      )
    elif column.type == ds_pb.ColumnType.CATEGORICAL_SET:
      dst_feature = tf.RaggedTensorSpec(
          shape=[None, None],
          dtype=tf_dtype,
      )
    else:
      raise ValueError(
          f"Unsupported semantic {column.type} for single-dim feature"
          f" {src_feature.name!r}"
      )
    input_signature[src_feature.name] = dst_feature

  return input_signature


def tensorflow_feature_spec(
    model: "generic_model.GenericModel",
    feature_dtypes: Dict[str, TFDType],
) -> Dict[str, Any]:
  """A TF feature spec used to deserialize tf example protos."""

  def missing_value(tf_dtype):
    """Representation of a missing value; if possible."""
    # Follow the missing value representation described in ydf.Semantic.
    if tf_dtype.is_floating:
      return math.nan
    elif tf_dtype == tf.string:
      return ""
    else:
      return None  # Missing value not allowed

  model_dataspec = model.data_spec()
  input_features = model.input_features()
  input_feature_names_set = set(f.name for f in input_features)

  feature_spec = {}

  # Multi-dim features
  for unstacked in model_dataspec.unstackeds:
    if unstacked.size == 0:
      raise RuntimeError("Empty unstacked")
    sub_names = dataset_io.unrolled_feature_names(
        unstacked.original_name, unstacked.size
    )
    # Note: The "input_features" contain unrolled feature names.
    if sub_names[0] not in input_feature_names_set:
      continue

    tf_dtype = tf_feature_dtype_manual(
        unstacked.original_name,
        unstacked.begin_column_idx,
        model_dataspec,
        feature_dtypes,
    )

    tfe_dtype = mapping_tf_dtype_to_tf_example_dtype().get(tf_dtype)
    if tfe_dtype is None:
      raise ValueError(f"Unsupported dtype: {tf_dtype}")

    if unstacked.type in [
        ds_pb.ColumnType.NUMERICAL,
        ds_pb.ColumnType.DISCRETIZED_NUMERICAL,
        ds_pb.ColumnType.CATEGORICAL,
        ds_pb.ColumnType.BOOLEAN,
    ]:
      effective_dtype = tfe_dtype
      effective_missing_value = missing_value(effective_dtype)
      if effective_missing_value is None:
        multidim_missing_value = None
      else:
        multidim_missing_value = [
            missing_value(effective_dtype)
        ] * unstacked.size

      dst_feature = tf.io.FixedLenFeature(
          shape=[unstacked.size],
          dtype=effective_dtype,
          default_value=multidim_missing_value,
      )
    else:
      raise ValueError(
          f"Unsupported semantic {unstacked.type} for multi-dim feature"
          f" {unstacked.original_name!r}"
      )
    feature_spec[unstacked.original_name] = dst_feature

  # Single-dim features
  for src_feature in input_features:
    column = model_dataspec.columns[src_feature.column_idx]
    if column.is_unstacked:
      continue

    tf_dtype = tf_feature_dtype(src_feature, model_dataspec, feature_dtypes)

    tfe_dtype = mapping_tf_dtype_to_tf_example_dtype().get(tf_dtype)
    if tfe_dtype is None:
      raise ValueError(f"Unsupported dtype: {tf_dtype}")

    if column.type in [
        ds_pb.ColumnType.NUMERICAL,
        ds_pb.ColumnType.DISCRETIZED_NUMERICAL,
        ds_pb.ColumnType.CATEGORICAL,
        ds_pb.ColumnType.BOOLEAN,
    ]:
      effective_dtype = tfe_dtype
      dst_feature = tf.io.FixedLenFeature(
          shape=[],
          dtype=effective_dtype,
          default_value=missing_value(effective_dtype),
      )
    elif src_feature.semantic == dataspec.Semantic.CATEGORICAL_SET:
      dst_feature = tf.io.RaggedFeature(
          dtype=tfe_dtype, row_splits_dtype=tf.dtypes.int64
      )
    else:
      raise ValueError(f"Unsupported semantic: {src_feature.semantic}")
    feature_spec[src_feature.name] = dst_feature

  return feature_spec


def _unroll_dict(
    src: Dict[str, TFTensor],
    data_spec: ds_pb.DataSpecification,
    input_features: Sequence[str],
) -> Dict[str, TFTensor]:
  """Unrolls multi-dimensional features.

  This function mirrors "_unroll_dict" in "dataset/io/dataset_io.py" which
  unrolls numpy arrays automatically (i.e., without a dataspec).

  Args:
    src: Dictionary of single and multi-dimensional values.
    data_spec: A dataspec.
    input_features: Input features to unroll.

  Returns:
    Dictionary containing only single-dimensional values.
  """

  try:
    dst = {}
    # Unroll multi-dim features.
    input_features_set = set(input_features)
    for unstacked in data_spec.unstackeds:
      sub_names = dataset_io.unrolled_feature_names(
          unstacked.original_name, unstacked.size
      )
      if sub_names[0] not in input_features_set:
        continue
      value = src[unstacked.original_name]
      for dim_idx, sub_name in enumerate(sub_names):
        dst[sub_name] = value[:, dim_idx]

    # Copy single-dim features
    for column in data_spec.columns:
      if column.is_unstacked:
        continue
      if column.name not in input_features_set:
        continue
      dst[column.name] = src[column.name]
    return dst
  except (KeyError, ValueError) as exc:
    exc.add_note(
        f"While looking for unrolled features {input_features!r} in the tensor"
        f" dictionary {src!r}"
    )
    raise exc
