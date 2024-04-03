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

"""An analysis contains information about a model."""

import copy
import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from yggdrasil_decision_forests.dataset import data_spec_pb2
from ydf.cc import ydf
from ydf.dataset import dataspec
from yggdrasil_decision_forests.utils import model_analysis_pb2
from yggdrasil_decision_forests.utils import partial_dependence_plot_pb2

Bin = (
    partial_dependence_plot_pb2.PartialDependencePlotSet.PartialDependencePlot.Bin
)
AttributeInfo = (
    partial_dependence_plot_pb2.PartialDependencePlotSet.PartialDependencePlot.AttributeInfo
)


@dataclasses.dataclass(frozen=True)
class PartialDependencePlot:
  """A partial dependence plot (PDP).

  Attributes:
    predictions: PDP value for each of the bins. If the model output is
      multi-dimensional, `predictions` is of shape [num bins, num dimensions].
      If the model output is single-dimensional, `predictions` is of shape [num
      bins].
    feature_names: Input feature names.
    feature_values: Input feature values. `feature_values[i][j]` is the value of
      feature `feature_names[i]` for the prediction `predictions[j]`.
  """

  feature_names: Sequence[str]
  feature_values: Sequence[np.ndarray]
  predictions: np.ndarray


class Analysis:
  """A model analysis.

  An analysis contains information about a model (e.g., variable
  importance, training logs), a dataset (e.g., column statistics), and the
  application of the model on this dataset (e.g. partial dependence plots).
  """

  def __init__(
      self,
      analysis_proto: model_analysis_pb2.StandaloneAnalysisResult,
      options_proto: model_analysis_pb2.Options,
  ):
    self._analysis_proto = analysis_proto
    self._options_proto = options_proto

  def __str__(self) -> str:
    """Returns the string representation of the analysis."""
    return (
        "A model analysis. Use a notebook cell to display the analysis."
        " Alternatively, export the analysis with"
        ' `analysis.to_file("analysis.html")`.'
    )

  def html(self) -> str:
    """Html representation of the analysis."""

    return self._repr_html_()

  def _repr_html_(self) -> str:
    """Returns the Html representation of the analysis."""

    effective_options = copy.deepcopy(self._options_proto)
    effective_options.report_header.enabled = False
    effective_options.report_setup.enabled = False
    effective_options.model_description.enabled = False
    effective_options.plot_width = 400
    effective_options.plot_height = 300
    effective_options.figure_width = effective_options.plot_width * 3

    return ydf.ModelAnalysisCreateHtmlReport(
        self._analysis_proto, effective_options
    )

  def to_file(self, path: str) -> None:
    """Exports the analysis to a html file."""
    with open(path, "w") as f:
      f.write(self.html())

  def partial_dependence_plots(self) -> Sequence[PartialDependencePlot]:
    """Programmatic access to the Partial Dependence Plots (PDP).

    A PDP is a powerful tool to interpret a model.
    See https://christophm.github.io/interpretable-ml-book/pdp.html

    Note: PDPs can be plotted automatically with `model.analyze`.

    Usage example:

    ```python
    import ydf
    import matplotlib.pyplot as plt

    # Get the model PDPs
    pdps = ydf.load_model(...).analyze(...).partial_dependence_plots()

    # Select the first PDP
    pdp = pdps[0]

    # Check the PDP is single dimensional.
    assert len(pdp.feature_names) == 1, "PDP not single dimensional"

    plt.plot(pdp.feature_values[0], pdp.predictions)
    plt.xlabel(pdp.feature_names[0])
    plt.ylabel("prediction")
    plt.show()
    ```
    """

    # Unfold the PDP protos into a python structure.
    dst_pdps = []
    for src_pdp in self._analysis_proto.core_analysis.pdp_set.pdps:

      # Note: If "_pdp_prediction_value" returns a numpy array, the following
      # "np.array" acts as a "np.stack".
      predictions = np.array([
          _pdp_prediction_value(bin, src_pdp.num_observations)
          for bin in src_pdp.pdp_bins
      ])

      feature_names = [
          _pdp_feature_name(attr, data_spec=self._analysis_proto.data_spec)
          for attr in src_pdp.attribute_info
      ]

      feature_values = []
      for local_attribute_idx, attr in enumerate(src_pdp.attribute_info):
        column_spec = self._analysis_proto.data_spec.columns[attr.attribute_idx]
        if column_spec.HasField("categorical"):
          categorical_dictionary = (
              dataspec.categorical_column_dictionary_to_list(column_spec)
          )
        else:
          categorical_dictionary = None

        feature_values.append(
            np.array([
                _pdp_feature_value(
                    bin, local_attribute_idx, categorical_dictionary
                )
                for bin in src_pdp.pdp_bins
            ])
        )

      dst_pdps.append(
          PartialDependencePlot(
              predictions=predictions,
              feature_values=feature_values,
              feature_names=feature_names,
          )
      )
    return dst_pdps

  def variable_importances(self) -> Dict[str, List[Tuple[float, str]]]:
    """Programmatic access to the Variable importances.

    Note: Variable importances can be plotted automatically with
    `model.analyze`.

    Usage example:

    ```python

    # Get the variable importances
    vis = ydf.load_model(...).analyze(...).variable_importances()

    # Print the variable importances
    print(vis)
    ```

    Returns:
      Variable importances.
    """

    def feature_name(attribute_idx: int) -> str:
      return self._analysis_proto.data_spec.columns[attribute_idx].name

    # Unfold the VIs protos into a python structure.
    variable_importances = {}
    for (
        name,
        importance_set,
    ) in self._analysis_proto.core_analysis.variable_importances.items():
      variable_importances[name] = [
          (src.importance, feature_name(src.attribute_idx))
          for src in importance_set.variable_importances
      ]
    return variable_importances


class PredictionAnalysis:
  """A prediction analysis.

  A prediction analysis explains why a model made a prediction.
  """

  def __init__(
      self,
      analysis_proto: model_analysis_pb2.PredictionAnalysisResult,
      options_proto: model_analysis_pb2.PredictionAnalysisOptions,
  ):
    self._analysis_proto = analysis_proto
    self._options_proto = options_proto

  def __str__(self) -> str:
    """Returns the string representation of the analysis."""
    return (
        "A prediction analysis. Use a notebook cell to display the analysis."
        " Alternatively, export the analysis with"
        ' `analysis.to_file("analysis.html")`.'
    )

  def html(self) -> str:
    """Html representation of the analysis."""

    return self._repr_html_()

  def _repr_html_(self) -> str:
    """Returns the Html representation of the analysis."""

    effective_options = copy.deepcopy(self._options_proto)
    return ydf.PredictionAnalysisCreateHtmlReport(
        self._analysis_proto, effective_options
    )

  def to_file(self, path: str) -> None:
    """Exports the analysis to a html file."""
    with open(path, "w") as f:
      f.write(self.html())


def _pdp_prediction_value(
    bin: Bin, num_observations: float
) -> Union[float, np.ndarray]:
  """Extracts a uniform numerical prediction value from a bin."""
  if bin.prediction.HasField("sum_of_regression_predictions"):
    return bin.prediction.sum_of_regression_predictions / num_observations
  elif bin.prediction.HasField("sum_of_ranking_predictions"):
    return bin.prediction.sum_of_ranking_predictions / num_observations
  elif bin.prediction.HasField("classification_class_distribution"):
    # Skip OOV item.
    return (
        np.array(bin.prediction.classification_class_distribution.counts[1:])
        / bin.prediction.classification_class_distribution.sum
    )
  else:
    raise ValueError(f"Unsupported prediction type: {bin}")


def _pdp_feature_name(
    attr: AttributeInfo, data_spec: data_spec_pb2.DataSpecification
) -> str:
  """Name of a feature."""
  return data_spec.columns[attr.attribute_idx].name


def _pdp_feature_value(
    bin: Bin,
    local_attribute_idx: int,
    categorical_dictionary: Optional[List[str]],
) -> Union[bool, float, int, bool, str]:
  """Value of a feature."""
  value = bin.center_input_feature_values[local_attribute_idx]
  if value.HasField("boolean"):
    return value.boolean
  elif value.HasField("numerical"):
    return value.numerical
  elif value.HasField("categorical"):
    assert categorical_dictionary is not None
    return categorical_dictionary[value.categorical]
  else:
    raise ValueError(f"Unsupported value: {value}")
