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

"""Standalone evaluation of models."""

from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from yggdrasil_decision_forests.metric import metric_pb2
from ydf.cc import ydf
from ydf.metric import metric
from ydf.model import generic_model
from ydf.utils import concurrency


def _build_evaluation_options(
    task: generic_model.Task,
    bootstrapping: Union[bool, int],
    ndcg_truncation: int,
    mrr_truncation: int,
    num_threads: Optional[int],
) -> metric_pb2.EvaluationOptions:
  """Builds evaluation options for the given task."""
  if isinstance(bootstrapping, bool):
    bootstrapping_samples = 2000 if bootstrapping else -1
  elif isinstance(bootstrapping, int) and bootstrapping >= 100:
    bootstrapping_samples = bootstrapping
  else:
    raise ValueError(
        "bootstrapping argument should be boolean or an integer greater than"
        " 100 as bootstrapping will not yield useful results otherwise. Got"
        f" {bootstrapping!r}"
    )
  ranking = None
  if task == generic_model.Task.RANKING:
    ranking = metric_pb2.EvaluationOptions.Ranking(
        ndcg_truncation=ndcg_truncation, mrr_truncation=mrr_truncation
    )
  options = metric_pb2.EvaluationOptions(
      bootstrapping_samples=bootstrapping_samples,
      task=task._to_proto_type(),  # pylint: disable=protected-access
      ranking=ranking,
      num_threads=num_threads,
  )
  return options


def _check_task(task: generic_model.Task) -> None:
  """Checks that the task is supported."""
  if task in (
      generic_model.Task.CLASSIFICATION,
      generic_model.Task.REGRESSION,
      generic_model.Task.RANKING,
  ):
    return
  elif task in (
      generic_model.Task.CATEGORICAL_UPLIFT,
      generic_model.Task.NUMERICAL_UPLIFT,
  ):
    raise ValueError("Uplift evaluation not supported.")
  elif task == generic_model.Task.ANOMALY_DETECTION:
    raise ValueError(
        "Anomaly detection models must be evaluated as binary classification."
    )
  else:
    raise ValueError(f"Unsupported task: {task}")


def _process_classification_labels(
    labels: np.ndarray, label_classes: Optional[List[str]]
) -> npt.NDArray[np.int32]:
  """Returns normalized labels and label classes (if not provided)."""
  if np.issubdtype(labels.dtype, np.integer):
    return labels.astype(np.int32)
  elif np.issubdtype(labels.dtype, np.floating):
    raise ValueError(
        "Floating-point labels are not supported, use integer or string labels."
    )
  else:
    try:
      labels = labels.astype(np.str_)
    except Exception as exc:
      raise ValueError("Could not convert labels to string") from exc
    if label_classes is None:
      raise ValueError(
          "When using string labels, label_classes must be provided"
      )
    if not all(isinstance(name, str) for name in label_classes):
      raise ValueError("Label classes must be strings.")

    string_to_position = {
        name: position for position, name in enumerate(label_classes)
    }

    def map_func(x):
      pos = string_to_position.get(x, None)
      if pos is None:
        raise ValueError(f"Found label not in label_classes: {x}")
      return pos

    labels = np.vectorize(map_func)(labels)
    return labels.astype(np.int32)


def _normalize_weights(
    weights: Optional[npt.NDArray[np.float32]],
    num_examples: int,
    options: metric_pb2.EvaluationOptions,
) -> npt.NDArray[np.float32]:
  """Processes weights and updates evaluation options."""
  if weights is not None:
    if weights.ndim != 1:
      raise ValueError(f"Weights must be a 1D array. Got: {weights.shape}.")
    if len(weights) != num_examples:
      raise ValueError(
          f"There must be one weight per example. Got: Weights: {len(weights)}"
          f" vs Examples: {num_examples}"
      )
    if np.issubdtype(weights.dtype, np.floating):
      weights = weights.astype(np.float32)
    else:
      raise ValueError(
          f"Weights must be floating point values, got {weights.dtype}"
      )
    options.weights.numerical.SetInParent()
  else:
    weights = np.array([], dtype=np.float32)
  return weights


def _normalize_ranking_groups(
    ranking_groups: Optional[npt.NDArray[np.uint64]],
    num_examples: int,
    task: generic_model.Task,
) -> Optional[npt.NDArray[np.uint64]]:
  """Processes ranking groups and checks if they are valid."""
  if ranking_groups is None:
    if task == generic_model.Task.RANKING:
      raise ValueError("Ranking group must be specified for ranking tasks.")
    ranking_groups = np.array([], dtype=np.uint64)
  else:
    if task != generic_model.Task.RANKING:
      raise ValueError(
          "Ranking groups must only be specified for ranking tasks."
      )
    if len(ranking_groups) != num_examples:
      raise ValueError(
          "There must be one ranking group per example. Got: Ranking groups:"
          f" {len(ranking_groups)} vs Examples: {num_examples}"
      )
    if np.issubdtype(ranking_groups.dtype, np.integer):
      ranking_groups = ranking_groups.astype(np.uint64)
    else:
      ranking_groups = np.array(
          [hash(item) % 2**64 for item in ranking_groups], dtype=np.uint64
      )
  return ranking_groups


def _process_classification_predictions(
    predictions: npt.NDArray[np.float32],
    num_examples: int,
) -> Tuple[npt.NDArray[np.float32], int]:
  """Returns formatted predictions and number of classes."""
  if predictions.ndim == 1:
    if len(predictions) != num_examples:
      raise ValueError(
          "There must be one prediction per example, got: Predictions:"
          f" {len(predictions)} vs Examples: {num_examples}"
      )
    return predictions.astype(np.float32), 2
  elif predictions.ndim == 2:
    if predictions.shape[0] != num_examples:
      raise ValueError(
          "There must be one prediction per example, got: Predictions:"
          f" {len(predictions)} vs Examples: {num_examples}"
      )
    num_classes = predictions.shape[1]
    if num_classes == 1:
      raise ValueError(
          "Classification probabilities should have shape [n] (for binary"
          f" classification) or [num_classes, n], got {predictions.shape}"
      )
    elif num_classes == 2:
      reshaped_predictions = predictions[:, 1]
      return reshaped_predictions.astype(np.float32), num_classes
    else:
      # Only a view
      reshaped_predictions = predictions.reshape(
          num_examples * num_classes, order="C"
      )
      return reshaped_predictions.astype(np.float32), num_classes
  else:
    raise ValueError(
        "Predictions must be 1- or 2-dimensional, got"
        f" {predictions.ndim} dimensions"
    )


def _process_regression_predictions_and_labels(
    predictions: npt.NDArray[np.float32],
    labels: npt.NDArray[np.float32],
    num_examples: int,
) -> npt.NDArray[np.float32]:
  """Check the format of the predictions and casts labels to float."""
  if predictions.ndim != 1:
    raise ValueError(
        "Predictions must be a 1D float array for regression tasks."
    )
  if len(predictions) != num_examples:
    raise ValueError(
        "There must be one prediction per example. Got: Predictions:"
        f" {len(predictions)} vs Examples: {num_examples}"
    )
  return labels.astype(np.float32)


def _process_ranking_predictions_and_labels(
    predictions: npt.NDArray[np.float32],
    labels: npt.NDArray[np.float32],
    num_examples: int,
) -> npt.NDArray[np.float32]:
  """Check the format of the predictions and casts labels to float."""
  if predictions.ndim != 1:
    raise ValueError("Predictions must be a 1D float array for ranking tasks.")
  if len(predictions) != num_examples:
    raise ValueError(
        "There must be one prediction per example. Got: Predictions:"
        f" {len(predictions)} vs Examples: {num_examples}"
    )
  return labels.astype(np.float32)


def evaluate_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: generic_model.Task,
    *,
    weights: Optional[np.ndarray] = None,
    label_classes: Optional[List[str]] = None,
    ranking_groups: Optional[np.ndarray] = None,
    bootstrapping: Union[bool, int] = False,
    ndcg_truncation: int = 5,
    mrr_truncation: int = 5,
    random_seed: int = 1234,
    num_threads: Optional[int] = None,
) -> metric.Evaluation:
  """Evaluates predictions against labels.

  This function allows to evaluate the predictions of any model (possibly
  non-ydf), against the labels with YDF's evaluation format.

  YDF models should be evaluated directly with `model.evaluate`, which is more
  efficient and convenient.

  For **binary classification** tasks, `predictions` should contain the
  predicted probabilities and should be of shape [n], or [n,2] where `n`
  is the number of examples. If `predictions` have shape `[n]`, they should
  contain the probability of the "positive" class. In the case `[n,2]`,
  `predicions[:0]`  and `predicions[:1]` should, respectively, be the
  probability of the "negative" and "positive" class. The labels should be a 1D
  array of shape [n], containing either integers 0 and 1, or strings. If the
  labels are strings, the `label_classes` must be provided with the "negative"
  class first. For integer labels, providing `label_classes` is optional and
  only used for display.

  For **multiclass classification** tasks, `predictions` should contain the
  predicted probabilities and should be of shape [n,k] where `n` is the number
  of examples. `predicions[:i]` should contain the probability of the `i`-th
  class. The labels should be a 1D integer array of shape [n].  The labels
  should be a 1D integer or string array of shape [n]. The names of the classes
  is given by `label_classes` in the same order as the predictions. If the
  labels are integers, they should be in the range 0, .., num_classes -1. If the
  labels are strings, `label_classes` must be provided.  For integer labels,
  providing `label_classes` is optional and only used for display.

  For **regression** tasks, `predictions` should contain the predicted values as
  a 1D float array of shape [n], where `n` is the number of examples. The labels
  should also be a 1D float array of shape [n].

  For **ranking** tasks, `predictions` should contain the predicted values as
  a 1D float array of shape [n], where `n` is the number of examples. The labels
  should also be a 1D float array of shape [n]. The ranking groups should be
  an integer array of shape [n].

  Uplift evaluations and anomaly detection evaluations are not supported.

  Usage examples:

  ```python
  from sklearn.linear_model import LogisticRegression
  import ydf

  X_train, X_test, y_train, y_test = ...  # Load data

  model = LogisticRegression()
  model.fit(X_train, y_train)
  predictions: np.ndarray = model.predict_proba(X_test)
  evaluation = ydf.evaluate.evaluate_predictions(
      predictions, y_test, ydf.Task.CLASSIFICATION
  )
  print(evaluation)
  evaluation  # Prints an interactive report in IPython / Colab notebooks.
  ```

  ```python
  import numpy as np
  import ydf

  predictions = np.linspace(0, 1, 100)
  labels = np.concatenate([np.ones(50), np.zeros(50)]).astype(float)
  evaluation = ydf.evaluate.evaluate_predictions(
      predictions, labels, ydf.Task.REGRESSIONS
  )
  print(evaluation)
  evaluation  # Prints an interactive report in IPython / Colab notebooks.
  ```

  Args:
    predictions: Array of predictions to evaluate. The "task" argument defines
      the expected shape of the prediction array.
    labels: Label values.The "task" argument defines the expected shape of the
      prediction array.
    task: Task of the model.
    weights: Weights of the examples as a 1D float array of shape [n]. If not
      provided, all examples have the idential weight.
    label_classes: Names of the labels. Only used for classification tasks.
    ranking_groups: Ranking groups as a 1D integer array of shape [n]. Only used
      for ranking tasks.
    bootstrapping: Controls whether bootstrapping is used to evaluate the
      confidence intervals and statistical tests (i.e., all the metrics ending
      with "[B]"). If set to false, bootstrapping is disabled. If set to true,
      bootstrapping is enabled and 2000 bootstrapping samples are used. If set
      to an integer, it specifies the number of bootstrapping samples to use. In
      this case, if the number is less than 100, an error is raised as
      bootstrapping will not yield useful results.
    ndcg_truncation: Controls at which ranking position the NDCG metric should
      be truncated. Default to 5. Ignored for non-ranking models.
    mrr_truncation: Controls at which ranking position the MRR metric loss
      should be truncated. Default to 5. Ignored for non-ranking models.
    random_seed: Random seed for sampling.
    num_threads: Number of threads used to run the model.

  Returns:
    Evaluation metrics.
  """
  _check_task(task)

  if num_threads is None:
    num_threads = concurrency.determine_optimal_num_threads(training=False)

  options = _build_evaluation_options(
      task,
      bootstrapping,
      ndcg_truncation,
      mrr_truncation,
      num_threads,
  )

  if labels.ndim != 1:
    raise ValueError(f"The labels must be a 1D array. Got: {labels.shape}.")
  num_examples = len(labels)

  if np.issubdtype(predictions.dtype, np.floating):
    predictions = predictions.astype(np.float32)
  else:
    raise ValueError(
        f"Predictions must be floating point values, got {predictions.dtype}"
    )

  if task == generic_model.Task.CLASSIFICATION:
    predictions, num_classes = _process_classification_predictions(
        predictions, num_examples
    )
    if label_classes is not None and len(label_classes) != num_classes:
      raise ValueError(
          "The number of label classes and the number of prediction dimensions"
          f" do not match. Got {len(label_classes)} label_classes and"
          f" {num_classes} prediction columns"
      )
    labels = _process_classification_labels(labels, label_classes)
    if label_classes is None:
      label_classes = [str(i) for i in range(num_classes)]
  elif task == generic_model.Task.REGRESSION:
    labels = _process_regression_predictions_and_labels(
        predictions, labels, num_examples
    )
  elif task == generic_model.Task.RANKING:
    labels = _process_ranking_predictions_and_labels(
        predictions, labels, num_examples
    )
  else:
    raise ValueError(f"Unsupported task: {task}")

  weights = _normalize_weights(weights, num_examples, options)
  ranking_groups = _normalize_ranking_groups(ranking_groups, num_examples, task)

  evaluation_proto = ydf.EvaluatePredictions(
      predictions,
      labels,
      options,
      weights,
      label_classes,
      ranking_groups,
      random_seed,
  )

  return metric.Evaluation(evaluation_proto)
