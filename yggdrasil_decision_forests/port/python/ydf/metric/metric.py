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

"""Metrics."""

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from yggdrasil_decision_forests.metric import metric_pb2
from ydf.dataset import dataspec
from ydf.utils import string_lib

# Offset added to skip the OOD item (always with index 0).
_OUT_OF_DICTIONARY_OFFSET = 1

# Confidence interval of a numerical metric.
ConfidenceInterval = Tuple[float, float]


@dataclasses.dataclass
class ConfusionMatrix:
  """A confusion matrix.

  See https://developers.google.com/machine-learning/glossary#confusion-matrix



  Attributes:
    classes: The label classes. The number of elements should match the size of
      `matrix`.
    matrix: A square matrix of size `len(classes)` x `len(classes)`. For
      unweighted evaluation, `matrix[i,j]` is the number of test examples with
      label `classes[i]` and predicted label `classes[j]`. For weighted
      evaluation, the confusion matrix contains sum of examples weights instead
      number of examples.
  """

  classes: Tuple[str, ...]
  matrix: npt.NDArray[np.float64]

  def __str__(self):
    return "label (row) \\ prediction (col)\n" + string_lib.table(
        content=self.matrix.tolist(),
        row_labels=self.classes,
        column_labels=self.classes,
    )

  def value(self, prediction_idx: int, label_idx: int) -> float:
    return self.matrix[label_idx, prediction_idx]


@dataclasses.dataclass
class CharacteristicPerThreshold:
  """Model recall, precision and other metrics per a threshold value.

  Only used for classification models.

  Attributes:
    threshold: Threshold.
    true_positive: True positive.
    false_positive: False positive.
    true_negative: True negative.
    false_negative: False negative.
  """

  threshold: float
  true_positive: float
  false_positive: float
  true_negative: float
  false_negative: float

  @property
  def recall(self) -> float:
    """Recall."""

    return _safe_div(
        self.true_positive, self.true_positive + self.false_negative
    )

  @property
  def specificity(self) -> float:
    """Specificity."""

    return _safe_div(
        self.true_negative, self.true_negative + self.false_positive
    )

  @property
  def false_positive_rate(self) -> float:
    """False positie rate."""

    return _safe_div(
        self.false_positive, self.false_positive + self.true_negative
    )

  @property
  def precision(self) -> float:
    """Precision."""

    return _safe_div(
        self.true_positive, self.true_positive + self.false_positive
    )

  @property
  def accuracy(self) -> float:
    """Accuracy."""

    return _safe_div(
        self.true_positive + self.true_negative,
        self.true_positive
        + self.false_positive
        + self.true_negative
        + self.false_negative,
    )


def _safe_div(a: float, b: float) -> float:
  """Returns a/b. If a==b==0, returns 0."""

  if b == 0:
    assert a == 0
    if a != 0:
      raise ValueError(
          f"Cannot divide a={a} by b={b}. If b==0, then a should be zero."
      )
    return 0
  return a / b


@dataclasses.dataclass
class Characteristic:
  """Model recall, precision and other metrics per threshold values.

  Only used for classification models.

  Attributes:
    name: Identifier of the characteristic.
    roc_auc: Area under the curve (AUC) of the Receiver operating characteristic
      (ROC) curve.
    pr_auc: Area under the curve (AUC) of the Precision-Recall curve.
    per_threshold: Model characteristics per thresholds.
    true_positives: List of true positives.
    false_positives: List of false positives.
    true_negatives: List of true negatives.
    false_negatives: List of false negatives.
    thresholds: List of threhsolds.
  """

  name: str
  roc_auc: float
  pr_auc: float
  per_threshold: List[CharacteristicPerThreshold]

  def __str__(self):
    return f"""name: {self.name}
ROC AUC: {self.roc_auc:g}
PR AUC: {self.pr_auc:g}
Num thresholds: {len(self.per_threshold)}
"""

  def __repr__(self):
    return self.__str__()

  @property
  def auc(self) -> float:
    """Alias for `roc_auc` i.e. the AUC of the ROC curve."""

    return self.roc_auc

  @property
  def true_positives(self) -> npt.NDArray[np.float32]:
    """List of true positives."""

    return np.array([t.true_positive for t in self.per_threshold], np.float32)

  @property
  def false_positives(self) -> npt.NDArray[np.float32]:
    """List of false positives."""

    return np.array([t.false_positive for t in self.per_threshold], np.float32)

  @property
  def true_negatives(self) -> npt.NDArray[np.float32]:
    """List of true negatives."""
    return np.array([t.true_negative for t in self.per_threshold], np.float32)

  @property
  def false_negatives(self) -> npt.NDArray[np.float32]:
    """List of true negatives."""

    return np.array([t.false_negative for t in self.per_threshold], np.float32)

  @property
  def thresholds(self) -> npt.NDArray[np.float32]:
    """List of thresholds."""

    return np.array([t.threshold for t in self.per_threshold], np.float32)

  @property
  def recalls(self) -> npt.NDArray[np.float32]:
    """List of recall."""

    return np.array(
        [t.recall for t in self.per_threshold],
        np.float32,
    )

  @property
  def specificities(self) -> npt.NDArray[np.float32]:
    """List of specificity."""

    return np.array(
        [t.specificity for t in self.per_threshold],
        np.float32,
    )

  @property
  def precisions(self) -> npt.NDArray[np.float32]:
    """List of precisions."""

    return np.array(
        [t.precision for t in self.per_threshold],
        np.float32,
    )

  @property
  def false_positive_rates(self) -> npt.NDArray[np.float32]:
    """List of false positive rates."""

    return np.array(
        [t.false_positive_rate for t in self.per_threshold],
        np.float32,
    )

  @property
  def accuracies(self) -> npt.NDArray[np.float32]:
    """List of accuracies."""

    return np.array(
        [t.accuracy for t in self.per_threshold],
        np.float32,
    )


@dataclasses.dataclass
class Evaluation:
  """A collection of metrics, plots and tables about the quality of a model.

  Basic usage example:

  ```python
  evaluation = ydf.metric.Evaluation()
  evaluation["accuracy"] = 0.6
  ```

  An evaluation can be constructed with constructor arguments:

  ```python
  evaluation = ydf.metric.Evaluation(accuracy=0.6, num_examples=10)
  ```

  An evaluation contains properties to easily access common metrics, as well as
  checks to make sure metrics are used correctly.

  ```python
  evaluation = ydf.metric.Evaluation()
  evaluation.accuracy = 0.6

  evaluation.accuracy = "hello"
  >> Warning: The "accuracy" is generally expected to be a float. Instead got a
  str.
  ```

  Attributes:
    loss: Model loss. The loss definition is model dependent.
    num_examples: Number of examples (non weighted).
    num_examples_weighted: Number of examples (with weight).
    accuracy:
    confusion_matrix:
    characteristics:
    rmse: Root Mean Square Error. Only available for regression task.
    rmse_ci95_bootstrap: 95% confidence interval of the RMSE computed using
      bootstrapping. Only available for regression task.
    ndcg:
    qini:
    auuc:
    custom_metrics: User custom metrics dictionary.
  """

  # Model generic
  loss: Optional[float] = None
  num_examples: Optional[int] = None
  num_examples_weighted: Optional[float] = None
  custom_metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)
  # Classification
  accuracy: Optional[float] = None
  confusion_matrix: Optional[ConfusionMatrix] = None
  characteristics: Optional[List[Characteristic]] = None
  # Regression
  rmse: Optional[float] = None
  rmse_ci95_bootstrap: Optional[ConfidenceInterval] = None
  # Ranking
  ndcg: Optional[float] = None
  # Uplift
  qini: Optional[float] = None
  auuc: Optional[float] = None

  def __str__(self) -> str:
    """Returns the string representation of an evaluation."""

    from ydf.metric import display_metric  # pylint: disable=g-import-not-at-top

    return display_metric.evaluation_to_str(self)

  def _repr_html_(self) -> str:
    """Html representation of the metrics."""

    from ydf.metric import display_metric  # pylint: disable=g-import-not-at-top

    return display_metric.evaluation_to_html_str(self)

  def to_dict(self) -> Dict[str, Any]:
    """Metrics in a dictionary."""

    output = {**self.custom_metrics}

    def add_item(key, value):
      if value is not None:
        output[key] = value

    add_item("loss", self.loss)
    add_item("num_examples", self.num_examples)
    add_item("num_examples_weighted", self.num_examples_weighted)
    add_item("accuracy", self.accuracy)
    add_item("confusion_matrix", self.confusion_matrix)

    if self.characteristics is not None:
      for idx, characteristic in enumerate(self.characteristics):
        base_name = f"characteristic_{idx}"
        add_item(f"{base_name}:name", characteristic.name)
        add_item(f"{base_name}:roc_auc", characteristic.roc_auc)
        add_item(f"{base_name}:pr_auc", characteristic.pr_auc)

    add_item("rmse", self.rmse)
    add_item("rmse_ci95_bootstrap", self.rmse_ci95_bootstrap)
    add_item("ndcg", self.ndcg)
    add_item("qini", self.qini)
    add_item("auuc", self.auuc)
    return output


def evaluation_proto_to_evaluation(
    src: metric_pb2.EvaluationResults,
) -> Evaluation:
  """Converts an evaluation from proto to python wrapper format.

  This function does not copy all the fields from the input evaluation proto.
  Instead, only metrics targeted as PYDF general users are exported. For
  instance, prediction samples are not exported.

  Currently, this function does not export characteristics (e.g. ROC curve) and
  confidence bounds.

  Metrics related to the out-of-dictionary (OOD) item in classification label
  column are not reported.

  Args:
    src: Evaluation in proto format.

  Returns:
    Evaluation object.
  """

  evaluation = Evaluation()

  if src.HasField("count_predictions_no_weight"):
    evaluation.num_examples = src.count_predictions_no_weight

  if src.HasField("count_predictions"):
    evaluation.num_examples_weighted = src.count_predictions

  if src.HasField("loss_value"):
    evaluation.loss = src.loss_value

  if src.HasField("classification"):
    classes = dataspec.categorical_column_dictionary_to_list(src.label_column)
    classes_wo_oov = classes[_OUT_OF_DICTIONARY_OFFSET:]

    if src.classification.HasField("confusion"):
      confusion = src.classification.confusion
      assert confusion.nrow == confusion.ncol, "Invalid confusion matrix"
      assert confusion.nrow == len(classes), "Invalid confusion matrix"
      assert confusion.nrow >= 1, "Invalid confusion matrix"
      raw_confusion = np.array(confusion.counts).reshape(
          confusion.nrow, confusion.nrow
      )

      evaluation.accuracy = safe_div(
          np.trace(raw_confusion), np.sum(raw_confusion)
      )

      evaluation.confusion_matrix = ConfusionMatrix(
          classes=tuple(classes_wo_oov),
          matrix=raw_confusion[
              _OUT_OF_DICTIONARY_OFFSET:, _OUT_OF_DICTIONARY_OFFSET:
          ],
      )

    if src.classification.rocs:
      characteristics = []
      for roc_idx, roc in enumerate(src.classification.rocs):
        if roc_idx == 0:
          # Skip the OOV item
          continue
        if roc_idx == 1 and len(src.classification.rocs) == 3:
          # In case of binary classification, skip the negative class
          continue
        name = f"'{classes[roc_idx]}' vs others"
        characteristics.append(
            Characteristic(
                name=name,
                roc_auc=roc.auc,
                pr_auc=roc.pr_auc,
                per_threshold=[
                    CharacteristicPerThreshold(
                        true_positive=x.tp,
                        false_positive=x.fp,
                        true_negative=x.tn,
                        false_negative=x.fn,
                        threshold=x.threshold,
                    )
                    for x in roc.curve
                ],
            )
        )
      evaluation.characteristics = characteristics

    if "loss" not in evaluation.to_dict() and src.classification.HasField(
        "sum_log_loss"
    ):
      evaluation.loss = src.classification.sum_log_loss / src.count_predictions

  if src.HasField("regression"):
    reg = src.regression
    if reg.HasField("sum_square_error"):
      # Note: The RMSE is not the empirical variance of the error i.e., there is
      # not corrective term to the denominator. This implementation is similar
      # to the ones in sciket-learn, tensorflow and ydf cc.
      evaluation.rmse = math.sqrt(
          safe_div(reg.sum_square_error, src.count_predictions)
      )

    if reg.HasField("bootstrap_rmse_lower_bounds_95p") and reg.HasField(
        "bootstrap_rmse_upper_bounds_95p"
    ):
      evaluation.rmse_ci95_bootstrap = (
          reg.bootstrap_rmse_lower_bounds_95p,
          reg.bootstrap_rmse_upper_bounds_95p,
      )

  if src.HasField("ranking"):
    rank = src.ranking
    if rank.HasField("ndcg"):
      evaluation.ndcg = rank.ndcg.value

  if src.HasField("uplift"):
    uplift = src.uplift
    if uplift.HasField("qini"):
      evaluation.qini = uplift.qini
    if uplift.HasField("auuc"):
      evaluation.auuc = uplift.auuc

  for k, v in src.user_metrics.items():
    evaluation.custom_metrics[k] = v

  return evaluation


def safe_div(a: float, b: float) -> float:
  """Returns a/b. If a==b==0, returns 0.

  If b==0 and a!=0, raises an exception.

  Args:
    a: Numerator.
    b: Denominator.
  """

  if b == 0.0:
    if a != 0.0:
      raise ValueError(
          f"Cannot divide `a={a}` by `b={b}`. If `b==0`, then `a` should be"
          " zero."
      )
    return 0.0
  return a / b
