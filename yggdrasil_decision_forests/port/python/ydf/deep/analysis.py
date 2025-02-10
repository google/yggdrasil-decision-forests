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

"""Model analysis for Deep Learning models."""

import bisect
import math
import time
from typing import Dict, List, Sequence, Tuple, Union

from absl import logging
import numpy as np

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.dataset import example_pb2
from ydf.dataset import dataset as dataset_lib
from ydf.dataset import dataspec as dataspec_lib
from ydf.dataset.io import dataset_io as dataset_io_lib
from ydf.dataset.io import generator as generator_lib
from ydf.model import analysis as analysis_lib
from ydf.model import generic_model
from ydf.utils import log
from yggdrasil_decision_forests.utils import distribution_pb2
from yggdrasil_decision_forests.utils import model_analysis_pb2
from yggdrasil_decision_forests.utils import partial_dependence_plot_pb2

PartialDependencePlotSet = partial_dependence_plot_pb2.PartialDependencePlotSet
PartialDependencePlot = PartialDependencePlotSet.PartialDependencePlot
AttributeInfo = PartialDependencePlot.AttributeInfo
VocabularyMapping = Dict[str, List[str]]


def _log_scale_better_than_uniform(bounds: np.ndarray):
  """Checks if a log scale looks better than a uniform scale."""
  if bounds.size == 0 or bounds.ndim != 1:
    return False
  margin = 0.1
  mid_value = bounds[len(bounds) // 2]
  res = (mid_value - bounds[0]) < margin * (bounds[-1] - bounds[0])
  return res


def _initialize_classification_prediction_proto(
    num_classes: int, proto: distribution_pb2.IntegerDistributionFloat
):
  """Initialize the proto field for an integer distribution."""
  assert num_classes > 0, f"num_classes must be positive, got {num_classes}"
  proto.counts.extend([0.0] * num_classes)
  proto.sum = 0.0


def _add_to_classification_prediction_proto(
    proto: distribution_pb2.IntegerDistributionFloat,
    val: Union[np.ndarray, np.floating, float],
):
  """Add a single vector to an integer distribution."""
  if not isinstance(val, np.ndarray) or len(val) == 1:
    if isinstance(val, np.ndarray):
      val = val[0]
    if len(proto.counts) != 3:
      raise ValueError(
          f"Invalid proto distribution {proto}, expected 3 dimensions"
      )
    proto.counts[1] += 1.0 - val
    proto.counts[2] += val
    proto.sum += 1.0
  else:
    if len(proto.counts) != len(val) + 1:
      raise ValueError(
          f"Invalid proto distribution {proto}, expected {len(val)} dimensions"
      )
    for idx, cur_val in enumerate(val):
      proto.counts[idx + 1] += cur_val
    proto.sum += 1


def get_pdp_features(
    input_features: Sequence[generic_model.InputFeature],
) -> Sequence[generic_model.InputFeature]:
  supported_semantics = [
      dataspec_lib.Semantic.CATEGORICAL,
      dataspec_lib.Semantic.NUMERICAL,
      dataspec_lib.Semantic.BOOLEAN,
  ]
  return [f for f in input_features if f.semantic in supported_semantics]


def get_numerical_bins(
    num_numerical_bins: int, spec_boundaries: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes the bin boundaries and centers from the spec boundaries.

  The two outer bins are centered on the outer boundaries to capture values not
  seen during training.
  The returned boundaries are left boundaries. The first interval has no left
  boundary (implicit -inf).

  Args:
    num_numerical_bins: The number of bins to create.
    spec_boundaries: The boundaries of the numerical feature as computed by the
      data spec.

  Returns:
    A tuple containing `num_numerical_bins` bin centers and
    `num_numerical_bins-1` boundaries of the bins.
  """
  if num_numerical_bins < 3:
    raise ValueError(
        f"There must be at least three numerical bins, got {num_numerical_bins}"
    )
  if len(spec_boundaries) < 2:
    raise ValueError(
        "There must be at least two boundaries in the dataspec, got"
        f" {len(spec_boundaries)}"
    )

  def norm_to_actual(x):
    norm_l = np.floor(x)
    norm_u = np.ceil(x)
    l = spec_boundaries[norm_l.astype(int)]
    u = spec_boundaries[norm_u.astype(int)]
    return (x - norm_l) * (u - l) + l

  num_boundaries = len(spec_boundaries)
  normalized_spec_boundaries = np.arange(num_boundaries)
  bin_width_normalized = normalized_spec_boundaries[-1] / (
      num_numerical_bins - 2
  )
  normalized_bin_boundaries = bin_width_normalized * np.arange(
      num_numerical_bins - 1
  )
  normalized_centers = np.concatenate([
      [normalized_bin_boundaries[0]],
      (np.convolve(normalized_bin_boundaries, np.ones(2), "valid") / 2),
      [normalized_bin_boundaries[-1]],
  ])
  centers = norm_to_actual(normalized_centers)
  bin_boundaries = norm_to_actual(normalized_bin_boundaries)
  return centers, bin_boundaries


def get_attribute_bin_idx(
    example: generator_lib.NumpyExampleBatch,
    data_spec: data_spec_pb2.DataSpecification,
    attr_info: AttributeInfo,
) -> int:
  """Returns bin index a given example falls into for feature attr_info."""
  attribute_idx = attr_info.attribute_idx
  assert attribute_idx >= 0 and attribute_idx < len(
      data_spec.columns
  ), f"Invalid attribute index {attribute_idx}"
  column_spec = data_spec.columns[attribute_idx]
  assert (
      column_spec.name in example
  ), f"Missing feature {column_spec.name} in {example}"
  # For multi-dimensional features, we might allow multiple dimensions here one
  # day.
  assert (
      len(example[column_spec.name]) == 1
  ), f"Expected one example per batch, got {len(example[column_spec.name])}"
  if column_spec.type == data_spec_pb2.NUMERICAL:
    if column_spec.numerical.min_value == column_spec.numerical.max_value:
      return 0
    numerical_value = example[column_spec.name][0]
    boundaries = attr_info.numerical_boundaries
    return bisect.bisect_right(boundaries, numerical_value)
  elif column_spec.type == data_spec_pb2.CATEGORICAL:
    category_name = example[column_spec.name][0]
    if isinstance(category_name, float) and math.isnan(category_name):
      return dataspec_lib.YDF_OOD_DICTIONARY_ITEM_IDX
    if category_name in column_spec.categorical.items:
      return column_spec.categorical.items[category_name].index
    return dataspec_lib.YDF_OOD_DICTIONARY_ITEM_IDX
  elif column_spec.type == data_spec_pb2.BOOLEAN:
    return int(example[column_spec.name][0])
  else:
    raise ValueError(
        "Unsupported type for PDPs:"
        f" {dataspec_lib.Semantic.from_proto_type(column_spec.type)}"
    )


def update_obs_counters(
    example: generator_lib.NumpyExampleBatch,
    attr_info: AttributeInfo,
    data_spec: data_spec_pb2.DataSpecification,
) -> None:
  attribute_bin_idx = get_attribute_bin_idx(example, data_spec, attr_info)
  attr_info.num_observations_per_bins[attribute_bin_idx] += 1


def generate_modified_examples(
    example: generator_lib.NumpyExampleBatch,
    pdp: PartialDependencePlot,
    attr_name_to_modify: str,
    attr_type: data_spec_pb2.ColumnType,
    vocabulary_mappings: VocabularyMapping,
) -> generator_lib.NumpyExampleBatch:
  """Generates example batch that represents all bins for given attribute."""
  modified_examples = {}
  bin_centers = [bin.center_input_feature_values for bin in pdp.pdp_bins]
  if not bin_centers or not bin_centers[0]:
    raise ValueError(
        f"Missing bins or bin centers for feature {attr_name_to_modify}"
    )
  assert all(len(val) == 1 for val in bin_centers)
  # If adding additional semantics, it might be useful to refactor
  # generate_bins, generate_modified_examples and get_attribute_bin_idx
  # as separate functions per semantics.
  if attr_type == data_spec_pb2.CATEGORICAL:
    assert bin_centers[0][0].HasField(
        "categorical"
    ), f"Missing categorical bin centers for {attr_name_to_modify}"
    assert (
        attr_name_to_modify in vocabulary_mappings
    ), f"Missing mapping for {attr_name_to_modify}"
    vocab_map = vocabulary_mappings[attr_name_to_modify]
    assert all(val[0].HasField("categorical") for val in bin_centers)
    bin_center_values = [vocab_map[val[0].categorical] for val in bin_centers]
  elif attr_type == data_spec_pb2.NUMERICAL:
    assert bin_centers[0][0].HasField(
        "numerical"
    ), f"Missing numerical bin centers for {attr_name_to_modify}"
    bin_center_values = [val[0].numerical for val in bin_centers]
  elif attr_type == data_spec_pb2.BOOLEAN:
    assert bin_centers[0][0].HasField(
        "boolean"
    ), f"Missing boolean bin centers for {attr_name_to_modify}"
    bin_center_values = [val[0].boolean for val in bin_centers]
  else:
    raise ValueError(f"Invalid bin center {bin_centers}")
  for attr_name in example:
    if attr_name == attr_name_to_modify:
      modified_examples[attr_name] = np.array(bin_center_values)
    else:
      val = example[attr_name][0]
      num_bins = len(pdp.pdp_bins)
      modified_examples[attr_name] = np.repeat(val, num_bins)
  return modified_examples


def update_bin_with_prediction(
    current_bin: PartialDependencePlot.Bin, prediction: Union[float, np.ndarray]
):
  accumulator = current_bin.prediction
  if accumulator.HasField("classification_class_distribution"):
    _add_to_classification_prediction_proto(
        accumulator.classification_class_distribution, prediction
    )
  elif accumulator.HasField("sum_of_regression_predictions"):
    accumulator.sum_of_regression_predictions += prediction
  else:
    raise ValueError(
        "PDP bin is not initialized for CLASSIFICATION or REGRESSION."
    )


def update_pdp_set(
    model: generic_model.GenericModel,
    example: generator_lib.NumpyExampleBatch,
    pdp_set: partial_dependence_plot_pb2.PartialDependencePlotSet,
    vocabulary_mappings: VocabularyMapping,
):
  """Updates all PDPs with a single example."""
  data_spec = model.data_spec()
  for pdp in pdp_set.pdps:
    assert len(pdp.attribute_info) == 1
    attr_info = pdp.attribute_info[0]
    attr_name = data_spec.columns[attr_info.attribute_idx].name
    attr_type = data_spec.columns[attr_info.attribute_idx].type
    update_obs_counters(example, attr_info, data_spec)
    modified_examples = generate_modified_examples(
        example, pdp, attr_name, attr_type, vocabulary_mappings
    )
    predictions = model.predict(modified_examples)
    for i, pdp_bin in enumerate(pdp.pdp_bins):
      update_bin_with_prediction(pdp_bin, predictions[i])
    pdp.num_observations += 1


def generate_bins(
    attr_info: AttributeInfo,
    data_spec: data_spec_pb2.DataSpecification,
    num_numerical_bins: int,
    output_bins: List[PartialDependencePlot.Bin],
) -> None:
  """Generates the bins for the current feature."""
  column_spec = data_spec.columns[attr_info.attribute_idx]
  if column_spec.type == data_spec_pb2.NUMERICAL:
    attr_info.num_bins_per_input_feature = num_numerical_bins
    attr_info.num_observations_per_bins.extend([0.0] * num_numerical_bins)
    if not column_spec.discretized_numerical:
      raise ValueError(
          f"Column {column_spec.name} is NUMERICAL but does not have"
          " discretization statistics in the data spec. Make sure this model"
          " been trained for deep learning."
      )
    data_spec_boundaries = np.array(
        column_spec.discretized_numerical.boundaries
    )
    centers, boundaries = get_numerical_bins(
        num_numerical_bins, data_spec_boundaries
    )
    attr_info.numerical_boundaries.extend(boundaries)
    if _log_scale_better_than_uniform(boundaries):
      attr_info.scale = AttributeInfo.LOG
    output_bins.extend([
        PartialDependencePlot.Bin(
            center_input_feature_values=[
                example_pb2.Example.Attribute(numerical=val)
            ]
        )
        for val in centers
    ])

  elif column_spec.type == data_spec_pb2.CATEGORICAL:
    assert (
        not column_spec.categorical.is_already_integerized
    ), "Integerized columns are not supported"
    num_values = column_spec.categorical.number_of_unique_values
    attr_info.num_bins_per_input_feature = num_values
    attr_info.num_observations_per_bins.extend([0] * num_values)
    output_bins.extend([
        PartialDependencePlot.Bin(
            center_input_feature_values=[
                example_pb2.Example.Attribute(categorical=i)
            ]
        )
        for i in range(num_values)
    ])
  elif column_spec.type == data_spec_pb2.BOOLEAN:
    attr_info.num_bins_per_input_feature = 2
    attr_info.num_observations_per_bins.extend([0] * 2)
    output_bins.extend([
        PartialDependencePlot.Bin(
            center_input_feature_values=[
                example_pb2.Example.Attribute(boolean=val)
            ]
        )
        for val in [False, True]
    ])
  else:
    raise ValueError(
        "Unsupported type for PDPs:"
        f" {dataspec_lib.Semantic.from_proto_type(column_spec.type)}"
    )


def initialize_bin_prediction_counters(
    bins: Sequence[PartialDependencePlot.Bin],
    label_spec: data_spec_pb2.Column,
    task: generic_model.Task,
):
  """Initialize the prediction counters of the given bins."""
  if task == generic_model.Task.CLASSIFICATION:
    assert (
        label_spec.type == data_spec_pb2.CATEGORICAL
    ), "Label spec is not categorical"
    assert label_spec.HasField("categorical"), "Label spec is not categorical"
    num_classes = label_spec.categorical.number_of_unique_values
    for current_bin in bins:
      _initialize_classification_prediction_proto(
          num_classes, current_bin.prediction.classification_class_distribution
      )
    pass
  elif task == generic_model.Task.REGRESSION:
    for current_bin in bins:
      current_bin.prediction.sum_of_regression_predictions = 0.0
  else:
    raise ValueError(f"Unsupported task: {task}")


def initialize_pdp(
    pdp: PartialDependencePlot,
    feature: generic_model.InputFeature,
    model: generic_model.GenericModel,
    num_numerical_bins: int,
) -> None:
  """Initializes a PDP for a given feature."""
  assert model.label_col_idx() >= 0, "A label is required for model analysis."
  data_spec = model.data_spec()
  label_col_spec = data_spec.columns[model.label_col_idx()]
  pdp.type = PartialDependencePlot.PDP
  attr_info = pdp.attribute_info.add()
  attr_info.attribute_idx = feature.column_idx
  bins: List[PartialDependencePlot.Bin] = []
  generate_bins(attr_info, data_spec, num_numerical_bins, bins)
  initialize_bin_prediction_counters(bins, label_col_spec, model.task())
  pdp.pdp_bins.extend(bins)


def compute_partial_dependence_plot_set(
    model: generic_model.GenericModel,
    data: dataset_lib.InputDataset,
    options: model_analysis_pb2.Options,
    pdp_set: PartialDependencePlotSet,
) -> partial_dependence_plot_pb2.PartialDependencePlotSet:
  """Computes the PDP set of a model."""
  pdp_features = get_pdp_features(model.input_features())
  for feature in pdp_features:
    pdp = pdp_set.pdps.add()
    initialize_pdp(
        pdp,
        feature,
        model,
        options.pdp.num_numerical_bins,
    )

  cutoff_time = None
  if options.HasField("maximum_duration_seconds"):
    cutoff_time = time.time() + options.maximum_duration_seconds

  ds_generator = dataset_io_lib.build_batched_example_generator(data)
  vocabulary_mapping = {
      col.name: dataspec_lib.categorical_column_dictionary_to_list(col)
      for col in model.data_spec().columns
  }
  num_examples_for_analysis = (
      ds_generator.num_examples * options.pdp.example_sampling
  )
  num_processed_examples = 0
  # TODO: Add tqdm
  # TODO: Expose random seed in the Python function signature.
  for batch_idx, raw_numpy_batch in enumerate(
      ds_generator.generate(
          batch_size=1, shuffle=True, seed=options.random_seed
      )
  ):
    if (batch_idx % 100) == 0:
      logging.log_every_n_seconds(
          logging.INFO, f"{batch_idx + 1} examples scanned.", 30
      )
    if cutoff_time is not None and time.time() > cutoff_time:
      log.info("Maximum duration reached. Interrupting analysis early.")
      break

    update_pdp_set(model, raw_numpy_batch, pdp_set, vocabulary_mapping)
    num_processed_examples += 1
    if num_processed_examples >= num_examples_for_analysis:
      break

  return pdp_set


def model_analysis(
    model: generic_model.GenericModel,
    data: dataset_lib.InputDataset,
    options: model_analysis_pb2.Options,
) -> analysis_lib.Analysis:
  """Analyses a YDF model in Python. Can be used for deep models."""
  result = model_analysis_pb2.StandaloneAnalysisResult(
      data_spec=model.data_spec(),
      task=model.task()._to_proto_type(),  # pylint:disable=protected-access
      label_col_idx=model.label_col_idx(),
  )
  core_analysis = result.core_analysis
  if options.pdp.enabled:
    compute_partial_dependence_plot_set(
        model, data, options, core_analysis.pdp_set
    )
  return analysis_lib.Analysis(result, options)
