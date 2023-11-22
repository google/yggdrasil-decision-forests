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

from ydf.cc import ydf
from yggdrasil_decision_forests.utils import model_analysis_pb2


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
