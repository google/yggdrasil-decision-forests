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

"""Configures the automatic selection of the input features."""

import copy
import dataclasses
import math
from typing import Dict, List, Optional, Tuple, Union
from ydf.dataset import dataset
from ydf.dataset import dataspec as dataspec_lib
from ydf.learner import abstract_feature_selector as abstract_feature_selector_lib
from ydf.learner import generic_learner as generic_learner_lib
from ydf.model import feature_selector_logs
from ydf.model import generic_model
from ydf.utils import log


@dataclasses.dataclass
class BackwardSelectionFeatureSelector(
    abstract_feature_selector_lib.AbstractFeatureSelector
):
  """Greedy backward feature selector.

  Attributes:
    removal_ratio: Ratio of the features are removed at each iteration. The
      smaller the value, the more accurate but the more expensive the results.
      Note that at lease one feature is removed at each round. Only one of
      `removal_ratio` and `removal_count` can be specified. If neither
      `removal_ratio` and `removal_count` is specified, `removal_ratio` defaults
      to 0.1.
    removal_count: Number of the features are removed at each iteration. The
      smaller the value, the more accurate but the more expensive the results.
      Only one of `removal_ratio` and `removal_count` can be specified.
    variable_importance: Name of the variable importance used to drive the
      feature selection. Should be one of the keys of
      `model.variable_importances()`. If None, a reasonable variable importance
      is selected automatically.
    allow_structural_variable_importance: If True, structural variable
      importances can be used to drive to the feature selection. If false and if
      only structural variable importances are available, raises an error.
    allow_model_self_evaluation_and_variable_importances: If True, and if the
      model produces a self evaluation and corresponding variable importances,
      use those values directly. If false, use the provided validation dataset
      to evaluate and compute the variable importances of the model.
  """

  removal_ratio: Optional[float] = None
  removal_count: Optional[int] = None
  variable_importance: Optional[str] = None
  allow_structural_variable_importance: bool = False
  allow_model_self_evaluation_and_variable_importances: bool = True

  def __post_init__(self):
    if self.removal_ratio is not None and self.removal_count is not None:
      raise ValueError(
          "Only one of removal_ratio or removal_count can be specified"
      )
    if self.removal_ratio is None and self.removal_count is None:
      # Apply default
      self.removal_ratio = 0.1

  def _get_variable_importance(
      self, variable_importances: Dict[str, List[Tuple[float, str]]]
  ) -> Tuple[List[Tuple[float, str]], str]:
    """Returns the variable importance to guide the feature selection."""

    if self.variable_importance is not None:
      # Return the user requested variable importance.
      if self.variable_importance not in variable_importances:
        raise ValueError(
            f"The variable importance {self.variable_importance} does not"
            " exist. The available variable importances are:"
            f" {variable_importances!r}"
        )
      return (
          variable_importances[self.variable_importance],
          self.variable_importance,
      )

    # Find a good variable importance
    sorted_candidate = sorted(list(variable_importances.keys()))
    for candidate_pattern in [
        "MEAN_DECREASE_IN_AUC_",
        "MEAN_DECREASE_IN_PRAUC_",
        "MEAN_DECREASE_IN_ACCURACY",
        "MEAN_DECREASE_IN_",
        "MEAN_INCREASE_IN_",
        "INV_MEAN_MIN_DEPTH",
    ]:
      for candidate in sorted_candidate:
        if candidate.startswith(candidate_pattern):
          return variable_importances[candidate], candidate

    raise ValueError("No valid variable importance found.")

  def _use_validation_for_training(
      self, learner: generic_learner_lib.GenericLearner
  ) -> bool:
    """Returns whether the validation dataset is used to train the model."""
    return learner.learner_name != "RANDOM_FOREST"

  def _use_self_evaluation(
      self,
      learner: generic_learner_lib.GenericLearner,
      valid: Optional[dataset.InputDataset] = None,
  ) -> bool:
    has_valid = valid is not None
    if learner.learner_name == "RANDOM_FOREST":
      has_self_va = learner.hyperparameters["compute_oob_variable_importances"]
      if has_valid:
        if has_self_va:
          raise ValueError(
              "The Random Forest learner is used both with"
              " `compute_oob_variable_importances=True` and a validation"
              " dataset. Only specify one of those."
          )
        else:
          use_self_evaluation = False
      else:
        use_self_evaluation = True
        if not has_self_va and not self.allow_structural_variable_importance:
          raise ValueError(
              "Out-of-bag variable importance computation is not enabled for"
              " the Random Forest learner"
              " (`compute_oob_variable_importances=False`; default value) and"
              " no validation dataset was provided. Therefore, only structural"
              " variable importances are available. While valid, this setup can"
              " lead to sub-optimal results. To fix this error, follow one of"
              " the following options: 1) Set "
              " `ydf.RandomForestLearner(compute_oob_variable_importances=True)`"
              " to enable the computation of OOB variable importance (slow,"
              " good results). 2) Call `learner.train(valid=...)` to provide a"
              " validation dataset (average speed, result dependent on"
              " problem). 3) Set"
              " `BackwardSelectionFeatureSelector(allow_structural_variable_importance=True)`"
              " to allow the use of structural variable importances (fast,"
              " results might be non optimal)."
          )
    elif learner.learner_name == "GRADIENT_BOOSTED_TREES":
      has_self_va = learner.hyperparameters[
          "compute_permutation_variable_importance"
      ]
      if has_valid:
        if has_self_va:
          use_self_evaluation = True
        else:
          use_self_evaluation = False
      else:
        use_self_evaluation = True
        if not has_self_va and not self.allow_structural_variable_importance:
          raise ValueError(
              "Permutation variable importance computation is not enabled for"
              " the Gradient Boosted Trees learner"
              " (`compute_permutation_variable_importance=False`; default"
              " value) and no validation dataset was provided. Therefore, only"
              " structural variable importances are available. While valid,"
              " this setup can lead to sub-optimal results. To fix this error,"
              " follow one of the following options: 1) Set"
              " `ydf.RandomForestLearner(compute_permutation_variable_importance=True)`"
              " to enable the computation of OOB variable importance (slow,"
              " good results). 2) Call `learner.train(valid=...)` to provide a"
              " validation dataset (average speed, result dependent on"
              " problem). 3) Set"
              " `BackwardSelectionFeatureSelector(allow_structural_variable_importance=True)`"
              " to allow the use of structural variable importances (fast,"
              " results might be non optimal)."
          )
    elif learner.learner_name == "CART":
      if has_valid:
        use_self_evaluation = False
      else:
        use_self_evaluation = True
        if not self.allow_structural_variable_importance:
          raise ValueError(
              "No validation dataset was provided to the CART learner."
              " Therefore, only structural variable importances are available."
              " While valid, this setup can lead to sub-optimal results. To fix"
              " this error, follow one of the following options: 1) Call"
              " `learner.train(valid=...)` to provide a validation dataset"
              " (average speed, result dependent on problem). 2) Set"
              " `BackwardSelectionFeatureSelector(allow_structural_variable_importance=True)`"
              " to allow the use of structural variable importances (fast,"
              " results might be non optimal)."
          )
    else:
      use_self_evaluation = not has_valid

    return use_self_evaluation

  def run(
      self,
      learner: generic_learner_lib.GenericLearner,
      ds: dataset.InputDataset,
      valid: Optional[dataset.InputDataset] = None,
      verbose: Optional[Union[int, bool]] = None,
  ) -> generic_model.ModelType:

    # TODO: Make it possible to load the dataset into a vertical dataset
    # and to reuse it at each iteration.

    # List the input features
    input_features = learner.extract_input_feature_names(ds)
    log.info(
        "Run backward feature selection on %d features", len(input_features)
    )

    use_self_evaluation = self._use_self_evaluation(learner, valid)

    logs = feature_selector_logs.FeatureSelectorLogs()

    best_input_features = None
    best_score = -math.inf
    best_evaluation_dict = None
    best_model = None

    # TODO: Add support for early stopping
    iteration_idx = 0
    current_input_features = input_features
    while current_input_features:

      # Train a model with the selected features
      log.info(
          "[Iteration %d] Train model on %d features",
          iteration_idx,
          len(current_input_features),
      )
      local_learner = copy.deepcopy(learner)
      local_learner._feature_selector = None  # pylint: disable=protected-access
      local_learner._data_spec_args.columns = (  # pylint: disable=protected-access
          dataspec_lib.normalize_column_defs(current_input_features)
      )
      # TODO: Add support for cross-validation.
      model = local_learner.train(
          ds=ds,
          valid=valid if self._use_validation_for_training(learner) else None,
          verbose=verbose if iteration_idx else log.reduce_verbose(verbose),
      )

      # Print the logs and store the results
      if use_self_evaluation:
        evaluation = model.self_evaluation()
      else:
        evaluation = model.evaluate(valid)

      # Filter out non-float metrics
      evaluation_dict = evaluation.to_dict()
      evaluation_dict = {
          k: v for k, v in evaluation_dict.items() if isinstance(v, float)
      }

      # Get the score metric to optimize
      score, score_key = self.get_objective_score(evaluation_dict)
      if iteration_idx == 0:
        log.info(
            'Optimizing metric "%s". The available metrics are %r',
            score_key,
            list(evaluation_dict),
        )
      log.info(
          "[Iteration %d] Score:%g Metrics:%s",
          iteration_idx,
          score,
          evaluation_dict,
      )
      logs.iterations.append(
          feature_selector_logs.Iteration(
              score=score,
              features=current_input_features,
              metrics=evaluation_dict,
          )
      )

      if score > best_score:
        best_score = score
        best_input_features = current_input_features
        best_evaluation_dict = evaluation_dict
        best_model = model
        logs.best_iteration_idx = iteration_idx

      # Compute the variable importance of each feature
      if use_self_evaluation:
        variable_importances = model.variable_importances()
      else:
        variable_importances = model.analyze(
            valid,
            partial_dependence_plot=False,
            conditional_expectation_plot=False,
            maximum_duration=None,
        ).variable_importances()

      selected_variable_importances, variable_importances_key = (
          self._get_variable_importance(variable_importances)
      )

      if iteration_idx == 0:
        log.info(
            'Guide feature selection using "%s" variable importance. The'
            " available variable importances are %r",
            variable_importances_key,
            list(variable_importances),
        )

      # Remove the worst features
      if self.removal_ratio is not None:
        num_to_remove = max(
            1,
            math.floor(len(selected_variable_importances) * self.removal_ratio),
        )
      elif self.removal_count is not None:
        num_to_remove = min(
            len(selected_variable_importances), max(1, self.removal_count)
        )
      else:
        assert False

      num_to_keep = len(selected_variable_importances) - num_to_remove
      current_input_features = [
          x[1] for x in selected_variable_importances[:num_to_keep]
      ]
      iteration_idx += 1

    if best_model is None:
      raise ValueError("No model was trained")

    log.info(
        "The best subset of features was found at"
        " iteration %d with score:%g, metrics:%s and %d/%d selected features",
        logs.best_iteration_idx,
        best_score,
        best_evaluation_dict,
        len(best_input_features),
        len(input_features),
    )

    best_model.set_feature_selection_logs(logs)
    return best_model
