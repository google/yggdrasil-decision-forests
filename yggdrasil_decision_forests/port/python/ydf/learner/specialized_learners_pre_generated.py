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

r"""Wrappers around the YDF learners.

This file is generated automatically by running the following commands:
  bazel build //external/ydf_cc/yggdrasil_decision_forests/port/python/ydf/learner:specialized_learners\
  && bazel-bin/external/ydf_cc/yggdrasil_decision_forests/port/python/ydf/learner/specialized_learners_generator\
  > external/ydf_cc/yggdrasil_decision_forests/port/python/ydf/learner/specialized_learners_pre_generated.py

Please don't change this file directly. Instead, changes the source. The
documentation source is contained in the "GetGenericHyperParameterSpecification"
method of each learner e.g. GetGenericHyperParameterSpecification in
learner/gradient_boosted_trees/gradient_boosted_trees.cc contains the
documentation (and meta-data) used to generate this file.

In particular, these pre-generated wrappers included in the source code are 
included for reference only. The actual wrappers are re-generated during
compilation.
"""

from typing import Dict, Optional, Sequence, Union

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from ydf.dataset import dataset
from ydf.dataset import dataspec
from ydf.learner import custom_loss
from ydf.learner import generic_learner
from ydf.learner import hyperparameters
from ydf.learner import tuner as tuner_lib
from ydf.model import generic_model
from ydf.model.gradient_boosted_trees_model import gradient_boosted_trees_model
from ydf.model.random_forest_model import random_forest_model


class RandomForestLearner(generic_learner.GenericLearner):
  r"""Random Forest learning algorithm.

  A Random Forest (https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
  is a collection of deep CART decision trees trained independently and without
  pruning. Each tree is trained on a random subset of the original training
  dataset (sampled with replacement).

  The algorithm is unique in that it is robust to overfitting, even in extreme
  cases e.g. when there are more features than training examples.

  It is probably the most well-known of the Decision Forest training
  algorithms.

  Usage example:

  ```python
  import ydf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")

  model = ydf.RandomForestLearner().train(dataset)

  print(model.summary())
  ```

  Hyperparameters are configured to give reasonable results for typical
  datasets. Hyperparameters can also be modified manually (see descriptions)
  below or by applying the hyperparameter templates available with
  `RandomForestLearner.hyperparameter_templates()` (see this function's
  documentation for
  details).

  Attributes:
    label: Label of the dataset. The label column should not be identified as a
      feature in the `features` parameter.
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT,
      Task.ANOMALY_DETECTION).
    weights: Name of a feature that identifies the weight of each example. If
      weights are not specified, unit weights are assumed. The weight column
      should not be identified as a feature in the `features` parameter.
    ranking_group: Only for `task=Task.RANKING`. Name of a feature that
      identifies queries in a query/document ranking task. The ranking group
      should not be identified as a feature in the `features` parameter.
    uplift_treatment: Only for `task=Task.CATEGORICAL_UPLIFT` and `task=Task`.
      NUMERICAL_UPLIFT. Name of a numerical feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment. Currently, only 0/1 binary treatments are supported.
    features: If None, all columns are used as features. The semantic of the
      features is determined automatically. Otherwise, if
      include_all_columns=False (default) only the column listed in `features`
      are imported. If include_all_columns=True, all the columns are imported as
      features and only the semantic of the columns NOT in `columns` is
      determined automatically. If specified,  defines the order of the features
      - any non-listed features are appended in-order after the specified
      features (if include_all_columns=True). The label, weights, uplift
      treatment and ranking_group columns should not be specified as features.
    include_all_columns: See `features`.
    max_vocab_count: Maximum size of the vocabulary of CATEGORICAL and
      CATEGORICAL_SET columns stored as strings. If more unique values exist,
      only the most frequent values are kept, and the remaining values are
      considered as out-of-vocabulary.
    min_vocab_frequency: Minimum number of occurrence of a value for CATEGORICAL
      and CATEGORICAL_SET columns. Value observed less than
      `min_vocab_frequency` are considered as out-of-vocabulary.
    discretize_numerical_columns: If true, discretize all the numerical columns
      before training. Discretized numerical columns are faster to train with,
      but they can have a negative impact on the model quality. Using
      `discretize_numerical_columns=True` is equivalent as setting the column
      semantic DISCRETIZED_NUMERICAL in the `column` argument. See the
      definition of DISCRETIZED_NUMERICAL for more details.
    num_discretized_numerical_bins: Number of bins used when disretizing
      numerical columns.
    max_num_scanned_rows_to_infer_semantic: Number of rows to scan when
      inferring the column's semantic if it is not explicitly specified. Only
      used when reading from file, in-memory datasets are always read in full.
      Setting this to a lower number will speed up dataset reading, but might
      result in incorrect column semantics. Set to -1 to scan the entire
      dataset.
    max_num_scanned_rows_to_compute_statistics: Number of rows to scan when
      computing a column's statistics. Only used when reading from file,
      in-memory datasets are always read in full. A column's statistics include
      the dictionary for categorical features and the mean / min / max for
      numerical features. Setting this to a lower number will speed up dataset
      reading, but skew statistics in the dataspec, which can hurt model quality
      (e.g. if an important category of a categorical feature is considered
      OOV). Set to -1 to scan the entire dataset.
    data_spec: Dataspec to be used (advanced). If a data spec is given,
      `columns`, `include_all_columns`, `max_vocab_count`,
      `min_vocab_frequency`, `discretize_numerical_columns` and
      `num_discretized_numerical_bins` will be ignored.
    adapt_bootstrap_size_ratio_for_maximum_training_duration: Control how the
      maximum training duration (if set) is applied. If false, the training stop
      when the time is used. If true, adapts the size of the sampled dataset
      used to train each tree such that `num_trees` will train within
      `maximum_training_duration`. Has no effect if there is no maximum training
      duration specified. Default: False.
    allow_na_conditions: If true, the tree training evaluates conditions of the
      type `X is NA` i.e. `X is missing`. Default: False.
    bootstrap_size_ratio: Number of examples used to train each trees; expressed
      as a ratio of the training dataset size. Default: 1.0.
    bootstrap_training_dataset: If true (default), each tree is trained on a
      separate dataset sampled with replacement from the original dataset. If
      false, all the trees are trained on the entire same dataset. If
      bootstrap_training_dataset:false, OOB metrics are not available.
        bootstrap_training_dataset=false is used in "Extremely randomized trees"
        (https://link.springer.com/content/pdf/10.1007%2Fs10994-006-6226-1.pdf).
      Default: True.
    categorical_algorithm: How to learn splits on categorical attributes. -
      `CART`: CART algorithm. Find categorical splits of the form "value \\in
      mask". The solution is exact for binary classification, regression and
      ranking. It is approximated for multi-class classification. This is a good
      first algorithm to use. In case of overfitting (very small dataset, large
      dictionary), the "random" algorithm is a good alternative. - `ONE_HOT`:
      One-hot encoding. Find the optimal categorical split of the form
      "attribute == param". This method is similar (but more efficient) than
      converting converting each possible categorical value into a boolean
      feature. This method is available for comparison purpose and generally
      performs worse than other alternatives. - `RANDOM`: Best splits among a
      set of random candidate. Find the a categorical split of the form "value
      \\in mask" using a random search. This solution can be seen as an
      approximation of the CART algorithm. This method is a strong alternative
      to CART. This algorithm is inspired from section "5.1 Categorical
      Variables" of "Random Forest", 2001.
        Default: "CART".
    categorical_set_split_greedy_sampling: For categorical set splits e.g.
      texts. Probability for a categorical value to be a candidate for the
      positive set. The sampling is applied once per node (i.e. not at every
      step of the greedy optimization). Default: 0.1.
    categorical_set_split_max_num_items: For categorical set splits e.g. texts.
      Maximum number of items (prior to the sampling). If more items are
      available, the least frequent items are ignored. Changing this value is
      similar to change the "max_vocab_count" before loading the dataset, with
      the following exception: With `max_vocab_count`, all the remaining items
      are grouped in a special Out-of-vocabulary item. With `max_num_items`,
      this is not the case. Default: -1.
    categorical_set_split_min_item_frequency: For categorical set splits e.g.
      texts. Minimum number of occurrences of an item to be considered.
      Default: 1.
    compute_oob_performances: If true, compute the Out-of-bag evaluation (then
      available in the summary and model inspector). This evaluation is a cheap
      alternative to cross-validation evaluation. Default: True.
    compute_oob_variable_importances: If true, compute the Out-of-bag feature
      importance (then available in the summary and model inspector). Note that
      the OOB feature importance can be expensive to compute. Default: False.
    growing_strategy: How to grow the tree. - `LOCAL`: Each node is split
      independently of the other nodes. In other words, as long as a node
      satisfy the splits "constraints (e.g. maximum depth, minimum number of
      observations), the node will be split. This is the "classical" way to grow
      decision trees. - `BEST_FIRST_GLOBAL`: The node with the best loss
      reduction among all the nodes of the tree is selected for splitting. This
      method is also called "best first" or "leaf-wise growth". See "Best-first
      decision tree learning", Shi and "Additive logistic regression : A
      statistical view of boosting", Friedman for more details. Default:
      "LOCAL".
    honest: In honest trees, different training examples are used to infer the
      structure and the leaf values. This regularization technique trades
      examples for bias estimates. It might increase or reduce the quality of
      the model. See "Generalized Random Forests", Athey et al. In this paper,
      Honest trees are trained with the Random Forest algorithm with a sampling
      without replacement. Default: False.
    honest_fixed_separation: For honest trees only i.e. honest=true. If true, a
      new random separation is generated for each tree. If false, the same
      separation is used for all the trees (e.g., in Gradient Boosted Trees
      containing multiple trees). Default: False.
    honest_ratio_leaf_examples: For honest trees only i.e. honest=true. Ratio of
      examples used to set the leaf values. Default: 0.5.
    in_split_min_examples_check: Whether to check the `min_examples` constraint
      in the split search (i.e. splits leading to one child having less than
      `min_examples` examples are considered invalid) or before the split search
      (i.e. a node can be derived only if it contains more than `min_examples`
      examples). If false, there can be nodes with less than `min_examples`
      training examples. Default: True.
    keep_non_leaf_label_distribution: Whether to keep the node value (i.e. the
      distribution of the labels of the training examples) of non-leaf nodes.
      This information is not used during serving, however it can be used for
      model interpretation as well as hyper parameter tuning. This can take lots
      of space, sometimes accounting for half of the model size. Default: True.
    max_depth: Maximum depth of the tree. `max_depth=1` means that all trees
      will be roots. `max_depth=-1` means that tree depth is not restricted by
      this parameter. Values <= -2 will be ignored. Default: 16.
    max_num_nodes: Maximum number of nodes in the tree. Set to -1 to disable
      this limit. Only available for `growing_strategy=BEST_FIRST_GLOBAL`.
      Default: None.
    maximum_model_size_in_memory_in_bytes: Limit the size of the model when
      stored in ram. Different algorithms can enforce this limit differently.
      Note that when models are compiled into an inference, the size of the
      inference engine is generally much smaller than the original model.
      Default: -1.0.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    mhld_oblique_max_num_attributes: For MHLD oblique splits i.e.
      `split_axis=MHLD_OBLIQUE`. Maximum number of attributes in the projection.
      Increasing this value increases the training time. Decreasing this value
      acts as a regularization. The value should be in [2,
      num_numerical_features]. If the value is above the total number of
      numerical features, the value is capped automatically. The value 1 is
      allowed but results in ordinary (non-oblique) splits. Default: None.
    mhld_oblique_sample_attributes: For MHLD oblique splits i.e.
      `split_axis=MHLD_OBLIQUE`. If true, applies the attribute sampling
      controlled by the "num_candidate_attributes" or
      "num_candidate_attributes_ratio" parameters. If false, all the attributes
      are tested. Default: None.
    min_examples: Minimum number of examples in a node. Default: 5.
    missing_value_policy: Method used to handle missing attribute values. -
      `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean
      (in case of numerical attribute) or the most-frequent-item (in case of
      categorical attribute) computed on the entire dataset (i.e. the
      information contained in the data spec). - `LOCAL_IMPUTATION`: Missing
      attribute values are imputed with the mean (numerical attribute) or
      most-frequent-item (in the case of categorical attribute) evaluated on the
      training examples in the current node. - `RANDOM_LOCAL_IMPUTATION`:
      Missing attribute values are imputed from randomly sampled values from the
      training examples in the current node. This method was proposed by Clinic
      et al. in "Random Survival Forests"
      (https://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908043).
        Default: "GLOBAL_IMPUTATION".
    num_candidate_attributes: Number of unique valid attributes tested for each
      node. An attribute is valid if it has at least a valid split. If
      `num_candidate_attributes=0`, the value is set to the classical default
      value for Random Forest: `sqrt(number of input attributes)` in case of
      classification and `number_of_input_attributes / 3` in case of regression.
      If `num_candidate_attributes=-1`, all the attributes are tested. Default:
      0.
    num_candidate_attributes_ratio: Ratio of attributes tested at each node. If
      set, it is equivalent to `num_candidate_attributes =
      number_of_input_features x num_candidate_attributes_ratio`. The possible
      values are between ]0, and 1] as well as -1. If not set or equal to -1,
      the `num_candidate_attributes` is used. Default: -1.0.
    num_oob_variable_importances_permutations: Number of time the dataset is
      re-shuffled to compute the permutation variable importances. Increasing
      this value increase the training time (if
      "compute_oob_variable_importances:true") as well as the stability of the
      oob variable importance metrics. Default: 1.
    num_trees: Number of individual decision trees. Increasing the number of
      trees can increase the quality of the model at the expense of size,
      training speed, and inference latency. Default: 300.
    pure_serving_model: Clear the model from any information that is not
      required for model serving. This includes debugging, model interpretation
      and other meta-data. The size of the serialized model can be reduced
      significatively (50% model size reduction is common). This parameter has
      no impact on the quality, serving speed or RAM usage of model serving.
      Default: False.
    random_seed: Random seed for the training of the model. Learners are
      expected to be deterministic by the random seed. Default: 123456.
    sampling_with_replacement: If true, the training examples are sampled with
      replacement. If false, the training samples are sampled without
      replacement. Only used when "bootstrap_training_dataset=true". If false
      (sampling without replacement) and if "bootstrap_size_ratio=1" (default),
      all the examples are used to train all the trees (you probably do not want
      that). Default: True.
    sorting_strategy: How are sorted the numerical features in order to find the
      splits - PRESORT: The features are pre-sorted at the start of the
      training. This solution is faster but consumes much more memory than
      IN_NODE. - IN_NODE: The features are sorted just before being used in the
      node. This solution is slow but consumes little amount of memory. .
      Default: "PRESORT".
    sparse_oblique_max_num_projections: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Maximum number of projections (applied after
      the num_projections_exponent). Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Increasing "max_num_projections" increases the training time but not the
      inference time. In late stage model development, if every bit of accuracy
      if important, increase this value. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) does not define this hyperparameter.
      Default: None.
    sparse_oblique_normalization: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before
      applying the sparse oblique projections. - `NONE`: No normalization. -
      `STANDARD_DEVIATION`: Normalize the feature by the estimated standard
      deviation on the entire train dataset. Also known as Z-Score
      normalization. - `MIN_MAX`: Normalize the feature by the range (i.e.
      max-min) estimated on the entire train dataset. Default: None.
    sparse_oblique_num_projections_exponent: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node. Increasing this value very likely improves the
      quality of the model, drastically increases the training time, and doe not
      impact the inference time. Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Therefore, increasing this `num_projections_exponent` and possibly
      `max_num_projections` may improve model quality, but will also
      significantly increase training time. Note that the complexity of
      (classic) Random Forests is roughly proportional to
      `num_projections_exponent=0.5`, since it considers sqrt(num_features) for
      a split. The complexity of (classic) GBDT is roughly proportional to
      `num_projections_exponent=1`, since it considers all features for a split.
      The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020)
      recommends values in [1/4, 2]. Default: None.
    sparse_oblique_projection_density_factor: Density of the projections as an
      exponent of the number of features. Independently for each projection,
      each feature has a probability "projection_density_factor / num_features"
      to be considered in the projection. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) calls this parameter `lambda` and
      recommends values in [1, 5]. Increasing this value increases training and
      inference time (on average). This value is best tuned for each dataset.
      Default: None.
    sparse_oblique_weights: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Possible values: - `BINARY`: The oblique
      weights are sampled in {-1,1} (default). - `CONTINUOUS`: The oblique
      weights are be sampled in [-1,1]. Default: None.
    split_axis: What structure of split to consider for numerical features. -
      `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This
      is the "classical" way to train a tree. Default value. - `SPARSE_OBLIQUE`:
      Sparse oblique splits (i.e. random splits one a small number of features)
      from "Sparse Projection Oblique Random Forests", Tomita et al., 2020. -
      `MHLD_OBLIQUE`: Multi-class Hellinger Linear Discriminant splits from
      "Classification Based on Multivariate Contrast Patterns", Canete-Sifuentes
      et al., 2029 Default: "AXIS_ALIGNED".
    uplift_min_examples_in_treatment: For uplift models only. Minimum number of
      examples per treatment in a node. Default: 5.
    uplift_split_score: For uplift models only. Splitter score i.e. score
      optimized by the splitters. The scores are introduced in "Decision trees
      for uplift modeling with single and multiple treatments", Rzepakowski et
      al. Notation: `p` probability / average value of the positive outcome, `q`
      probability / average value in the control group. - `KULLBACK_LEIBLER` or
      `KL`: - p log (p/q) - `EUCLIDEAN_DISTANCE` or `ED`: (p-q)^2 -
      `CHI_SQUARED` or `CS`: (p-q)^2/q
        Default: "KULLBACK_LEIBLER".
    winner_take_all: Control how classification trees vote. If true, each tree
      votes for one class. If false, each tree vote for a distribution of
      classes. winner_take_all_inference=false is often preferable. Default:
      True.
    num_threads: Number of threads used to train the model. Different learning
      algorithms use multi-threading differently and with different degree of
      efficiency. If `None`, `num_threads` will be automatically set to the
      number of processors (up to a maximum of 32; or set to 6 if the number of
      processors is not available). Making `num_threads` significantly larger
      than the number of processors can slow-down the training speed. The
      default value logic might change in the future.
    resume_training: If true, the model training resumes from the checkpoint
      stored in the `working_dir` directory. If `working_dir` does not contain
      any model checkpoint, the training starts from the beginning. Resuming
      training is useful in the following situations: (1) The training was
      interrupted by the user (e.g. ctrl+c or "stop" button in a notebook) or
      rescheduled, or (2) the hyper-parameter of the learner was changed e.g.
      increasing the number of trees.
    working_dir: Path to a directory available for the learning algorithm to
      store intermediate computation results. Depending on the learning
      algorithm and parameters, the working_dir might be optional, required, or
      ignored. For instance, distributed training algorithm always need a
      "working_dir", and the gradient boosted tree and hyper-parameter tuners
      will export artefacts to the "working_dir" if provided.
    resume_training_snapshot_interval_seconds: Indicative number of seconds in
      between snapshots when `resume_training=True`. Might be ignored by some
      learners.
    tuner: If set, automatically select the best hyperparameters using the
      provided tuner. When using distributed training, the tuning is
      distributed.
    workers: If set, enable distributed training. "workers" is the list of IP
      addresses of the workers. A worker is a process running
      `ydf.start_worker(port)`.
  """

  def __init__(
      self,
      label: str,
      task: generic_learner.Task = generic_learner.Task.CLASSIFICATION,
      weights: Optional[str] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      features: dataspec.ColumnDefs = None,
      include_all_columns: bool = False,
      max_vocab_count: int = 2000,
      min_vocab_frequency: int = 5,
      discretize_numerical_columns: bool = False,
      num_discretized_numerical_bins: int = 255,
      max_num_scanned_rows_to_infer_semantic: int = 10000,
      max_num_scanned_rows_to_compute_statistics: int = 10000,
      data_spec: Optional[data_spec_pb2.DataSpecification] = None,
      adapt_bootstrap_size_ratio_for_maximum_training_duration: Optional[
          bool
      ] = False,
      allow_na_conditions: Optional[bool] = False,
      bootstrap_size_ratio: Optional[float] = 1.0,
      bootstrap_training_dataset: Optional[bool] = True,
      categorical_algorithm: Optional[str] = "CART",
      categorical_set_split_greedy_sampling: Optional[float] = 0.1,
      categorical_set_split_max_num_items: Optional[int] = -1,
      categorical_set_split_min_item_frequency: Optional[int] = 1,
      compute_oob_performances: Optional[bool] = True,
      compute_oob_variable_importances: Optional[bool] = False,
      growing_strategy: Optional[str] = "LOCAL",
      honest: Optional[bool] = False,
      honest_fixed_separation: Optional[bool] = False,
      honest_ratio_leaf_examples: Optional[float] = 0.5,
      in_split_min_examples_check: Optional[bool] = True,
      keep_non_leaf_label_distribution: Optional[bool] = True,
      max_depth: Optional[int] = 16,
      max_num_nodes: Optional[int] = None,
      maximum_model_size_in_memory_in_bytes: Optional[float] = -1.0,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      mhld_oblique_max_num_attributes: Optional[int] = None,
      mhld_oblique_sample_attributes: Optional[bool] = None,
      min_examples: Optional[int] = 5,
      missing_value_policy: Optional[str] = "GLOBAL_IMPUTATION",
      num_candidate_attributes: Optional[int] = 0,
      num_candidate_attributes_ratio: Optional[float] = -1.0,
      num_oob_variable_importances_permutations: Optional[int] = 1,
      num_trees: Optional[int] = 300,
      pure_serving_model: Optional[bool] = False,
      random_seed: Optional[int] = 123456,
      sampling_with_replacement: Optional[bool] = True,
      sorting_strategy: Optional[str] = "PRESORT",
      sparse_oblique_max_num_projections: Optional[int] = None,
      sparse_oblique_normalization: Optional[str] = None,
      sparse_oblique_num_projections_exponent: Optional[float] = None,
      sparse_oblique_projection_density_factor: Optional[float] = None,
      sparse_oblique_weights: Optional[str] = None,
      split_axis: Optional[str] = "AXIS_ALIGNED",
      uplift_min_examples_in_treatment: Optional[int] = 5,
      uplift_split_score: Optional[str] = "KULLBACK_LEIBLER",
      winner_take_all: Optional[bool] = True,
      num_threads: Optional[int] = None,
      working_dir: Optional[str] = None,
      resume_training: bool = False,
      resume_training_snapshot_interval_seconds: int = 1800,
      tuner: Optional[tuner_lib.AbstractTuner] = None,
      workers: Optional[Sequence[str]] = None,
  ):

    hyper_parameters = {
        "adapt_bootstrap_size_ratio_for_maximum_training_duration": (
            adapt_bootstrap_size_ratio_for_maximum_training_duration
        ),
        "allow_na_conditions": allow_na_conditions,
        "bootstrap_size_ratio": bootstrap_size_ratio,
        "bootstrap_training_dataset": bootstrap_training_dataset,
        "categorical_algorithm": categorical_algorithm,
        "categorical_set_split_greedy_sampling": (
            categorical_set_split_greedy_sampling
        ),
        "categorical_set_split_max_num_items": (
            categorical_set_split_max_num_items
        ),
        "categorical_set_split_min_item_frequency": (
            categorical_set_split_min_item_frequency
        ),
        "compute_oob_performances": compute_oob_performances,
        "compute_oob_variable_importances": compute_oob_variable_importances,
        "growing_strategy": growing_strategy,
        "honest": honest,
        "honest_fixed_separation": honest_fixed_separation,
        "honest_ratio_leaf_examples": honest_ratio_leaf_examples,
        "in_split_min_examples_check": in_split_min_examples_check,
        "keep_non_leaf_label_distribution": keep_non_leaf_label_distribution,
        "max_depth": max_depth,
        "max_num_nodes": max_num_nodes,
        "maximum_model_size_in_memory_in_bytes": (
            maximum_model_size_in_memory_in_bytes
        ),
        "maximum_training_duration_seconds": maximum_training_duration_seconds,
        "mhld_oblique_max_num_attributes": mhld_oblique_max_num_attributes,
        "mhld_oblique_sample_attributes": mhld_oblique_sample_attributes,
        "min_examples": min_examples,
        "missing_value_policy": missing_value_policy,
        "num_candidate_attributes": num_candidate_attributes,
        "num_candidate_attributes_ratio": num_candidate_attributes_ratio,
        "num_oob_variable_importances_permutations": (
            num_oob_variable_importances_permutations
        ),
        "num_trees": num_trees,
        "pure_serving_model": pure_serving_model,
        "random_seed": random_seed,
        "sampling_with_replacement": sampling_with_replacement,
        "sorting_strategy": sorting_strategy,
        "sparse_oblique_max_num_projections": (
            sparse_oblique_max_num_projections
        ),
        "sparse_oblique_normalization": sparse_oblique_normalization,
        "sparse_oblique_num_projections_exponent": (
            sparse_oblique_num_projections_exponent
        ),
        "sparse_oblique_projection_density_factor": (
            sparse_oblique_projection_density_factor
        ),
        "sparse_oblique_weights": sparse_oblique_weights,
        "split_axis": split_axis,
        "uplift_min_examples_in_treatment": uplift_min_examples_in_treatment,
        "uplift_split_score": uplift_split_score,
        "winner_take_all": winner_take_all,
    }

    data_spec_args = dataspec.DataSpecInferenceArgs(
        columns=dataspec.normalize_column_defs(features),
        include_all_columns=include_all_columns,
        max_vocab_count=max_vocab_count,
        min_vocab_frequency=min_vocab_frequency,
        discretize_numerical_columns=discretize_numerical_columns,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        max_num_scanned_rows_to_infer_semantic=max_num_scanned_rows_to_infer_semantic,
        max_num_scanned_rows_to_compute_statistics=max_num_scanned_rows_to_compute_statistics,
    )

    deployment_config = self._build_deployment_config(
        num_threads=num_threads,
        resume_training=resume_training,
        resume_training_snapshot_interval_seconds=resume_training_snapshot_interval_seconds,
        working_dir=working_dir,
        workers=workers,
    )

    super().__init__(
        learner_name="RANDOM_FOREST",
        task=task,
        label=label,
        weights=weights,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        data_spec_args=data_spec_args,
        data_spec=data_spec,
        hyper_parameters=hyper_parameters,
        deployment_config=deployment_config,
        tuner=tuner,
    )

  def train(
      self,
      ds: dataset.InputDataset,
      valid: Optional[dataset.InputDataset] = None,
      verbose: Optional[Union[int, bool]] = None,
  ) -> random_forest_model.RandomForestModel:
    """Trains a model on the given dataset.

    Options for dataset reading are given on the learner. Consult the
    documentation of the learner or ydf.create_vertical_dataset() for additional
    information on dataset reading in YDF.

    Usage example:

    ```
    import ydf
    import pandas as pd

    train_ds = pd.read_csv(...)

    learner = ydf.RandomForestLearner(label="label")
    model = learner.train(train_ds)
    print(model.summary())
    ```

    If training is interrupted (for example, by interrupting the cell execution
    in Colab), the model will be returned to the state it was in at the moment
    of interruption.

    Args:
      ds: Training dataset.
      valid: Optional validation dataset. Some learners, such as Random Forest,
        do not need validation dataset. Some learners, such as
        GradientBoostedTrees, automatically extract a validation dataset from
        the training dataset if the validation dataset is not provided.
      verbose: Verbose level during training. If None, uses the global verbose
        level of `ydf.verbose`. Levels are: 0 of False: No logs, 1 or True:
        Print a few logs in a notebook; prints all the logs in a terminal. 2:
        Prints all the logs on all surfaces.

    Returns:
      A trained model.
    """
    return super().train(ds=ds, valid=valid, verbose=verbose)

  @classmethod
  def capabilities(cls) -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
        support_max_training_duration=True,
        resume_training=False,
        support_validation_dataset=False,
        support_partial_cache_dataset_format=False,
        support_max_model_size_in_memory=True,
        support_monotonic_constraints=False,
    )

  @classmethod
  def hyperparameter_templates(
      cls,
  ) -> Dict[str, hyperparameters.HyperparameterTemplate]:
    r"""Hyperparameter templates for this Learner.

    Hyperparameter templates are sets of pre-defined hyperparameters for easy
    access to different variants of the learner. Each template is a mapping to a
    set of hyperparameters and can be applied directly on the learner.

    Usage example:
    ```python
    templates = ydf.RandomForestLearner.hyperparameter_templates()
    better_defaultv1 = templates["better_defaultv1"]
    # Print a description of the template
    print(better_defaultv1.description)
    # Apply the template's settings on the learner.
    learner = ydf.RandomForestLearner(label, **better_defaultv1)
    ```

    Returns:
      Dictionary of the available templates
    """
    return {
        "better_defaultv1": hyperparameters.HyperparameterTemplate(
            name="better_default",
            version=1,
            description=(
                "A configuration that is generally better than the default"
                " parameters without being more expensive."
            ),
            parameters={"winner_take_all": True},
        ),
        "benchmark_rank1v1": hyperparameters.HyperparameterTemplate(
            name="benchmark_rank1",
            version=1,
            description=(
                "Top ranking hyper-parameters on our benchmark slightly"
                " modified to run in reasonable time."
            ),
            parameters={
                "winner_take_all": True,
                "categorical_algorithm": "RANDOM",
                "split_axis": "SPARSE_OBLIQUE",
                "sparse_oblique_normalization": "MIN_MAX",
                "sparse_oblique_num_projections_exponent": 1.0,
            },
        ),
    }


class GradientBoostedTreesLearner(generic_learner.GenericLearner):
  r"""Gradient Boosted Trees learning algorithm.

  A [Gradient Boosted Trees](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
  (GBT), also known as Gradient Boosted Decision Trees (GBDT) or Gradient
  Boosted Machines (GBM),  is a set of shallow decision trees trained
  sequentially. Each tree is trained to predict and then "correct" for the
  errors of the previously trained trees (more precisely each tree predict the
  gradient of the loss relative to the model output).

  Usage example:

  ```python
  import ydf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")

  model = ydf.GradientBoostedTreesLearner().train(dataset)

  print(model.summary())
  ```

  Hyperparameters are configured to give reasonable results for typical
  datasets. Hyperparameters can also be modified manually (see descriptions)
  below or by applying the hyperparameter templates available with
  `GradientBoostedTreesLearner.hyperparameter_templates()` (see this function's
  documentation for
  details).

  Attributes:
    label: Label of the dataset. The label column should not be identified as a
      feature in the `features` parameter.
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT).
    weights: Name of a feature that identifies the weight of each example. If
      weights are not specified, unit weights are assumed. The weight column
      should not be identified as a feature in the `features` parameter.
    ranking_group: Only for `task=Task.RANKING`. Name of a feature that
      identifies queries in a query/document ranking task. The ranking group
      should not be identified as a feature in the `features` parameter.
    uplift_treatment: Only for `task=Task.CATEGORICAL_UPLIFT` and `task=Task`.
      NUMERICAL_UPLIFT. Name of a numerical feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment. Currently, only 0/1 binary treatments are supported.
    features: If None, all columns are used as features. The semantic of the
      features is determined automatically. Otherwise, if
      include_all_columns=False (default) only the column listed in `features`
      are imported. If include_all_columns=True, all the columns are imported as
      features and only the semantic of the columns NOT in `columns` is
      determined automatically. If specified,  defines the order of the features
      - any non-listed features are appended in-order after the specified
      features (if include_all_columns=True). The label, weights, uplift
      treatment and ranking_group columns should not be specified as features.
    include_all_columns: See `features`.
    max_vocab_count: Maximum size of the vocabulary of CATEGORICAL and
      CATEGORICAL_SET columns stored as strings. If more unique values exist,
      only the most frequent values are kept, and the remaining values are
      considered as out-of-vocabulary.
    min_vocab_frequency: Minimum number of occurrence of a value for CATEGORICAL
      and CATEGORICAL_SET columns. Value observed less than
      `min_vocab_frequency` are considered as out-of-vocabulary.
    discretize_numerical_columns: If true, discretize all the numerical columns
      before training. Discretized numerical columns are faster to train with,
      but they can have a negative impact on the model quality. Using
      `discretize_numerical_columns=True` is equivalent as setting the column
      semantic DISCRETIZED_NUMERICAL in the `column` argument. See the
      definition of DISCRETIZED_NUMERICAL for more details.
    num_discretized_numerical_bins: Number of bins used when disretizing
      numerical columns.
    max_num_scanned_rows_to_infer_semantic: Number of rows to scan when
      inferring the column's semantic if it is not explicitly specified. Only
      used when reading from file, in-memory datasets are always read in full.
      Setting this to a lower number will speed up dataset reading, but might
      result in incorrect column semantics. Set to -1 to scan the entire
      dataset.
    max_num_scanned_rows_to_compute_statistics: Number of rows to scan when
      computing a column's statistics. Only used when reading from file,
      in-memory datasets are always read in full. A column's statistics include
      the dictionary for categorical features and the mean / min / max for
      numerical features. Setting this to a lower number will speed up dataset
      reading, but skew statistics in the dataspec, which can hurt model quality
      (e.g. if an important category of a categorical feature is considered
      OOV). Set to -1 to scan the entire dataset.
    data_spec: Dataspec to be used (advanced). If a data spec is given,
      `columns`, `include_all_columns`, `max_vocab_count`,
      `min_vocab_frequency`, `discretize_numerical_columns` and
      `num_discretized_numerical_bins` will be ignored.
    adapt_subsample_for_maximum_training_duration: Control how the maximum
      training duration (if set) is applied. If false, the training stop when
      the time is used. If true, the size of the sampled datasets used train
      individual trees are adapted dynamically so that all the trees are trained
      in time. Default: False.
    allow_na_conditions: If true, the tree training evaluates conditions of the
      type `X is NA` i.e. `X is missing`. Default: False.
    apply_link_function: If true, applies the link function (a.k.a. activation
      function), if any, before returning the model prediction. If false,
      returns the pre-link function model output. For example, in the case of
      binary classification, the pre-link function output is a logic while the
      post-link function is a probability. Default: True.
    categorical_algorithm: How to learn splits on categorical attributes. -
      `CART`: CART algorithm. Find categorical splits of the form "value \\in
      mask". The solution is exact for binary classification, regression and
      ranking. It is approximated for multi-class classification. This is a good
      first algorithm to use. In case of overfitting (very small dataset, large
      dictionary), the "random" algorithm is a good alternative. - `ONE_HOT`:
      One-hot encoding. Find the optimal categorical split of the form
      "attribute == param". This method is similar (but more efficient) than
      converting converting each possible categorical value into a boolean
      feature. This method is available for comparison purpose and generally
      performs worse than other alternatives. - `RANDOM`: Best splits among a
      set of random candidate. Find the a categorical split of the form "value
      \\in mask" using a random search. This solution can be seen as an
      approximation of the CART algorithm. This method is a strong alternative
      to CART. This algorithm is inspired from section "5.1 Categorical
      Variables" of "Random Forest", 2001.
        Default: "CART".
    categorical_set_split_greedy_sampling: For categorical set splits e.g.
      texts. Probability for a categorical value to be a candidate for the
      positive set. The sampling is applied once per node (i.e. not at every
      step of the greedy optimization). Default: 0.1.
    categorical_set_split_max_num_items: For categorical set splits e.g. texts.
      Maximum number of items (prior to the sampling). If more items are
      available, the least frequent items are ignored. Changing this value is
      similar to change the "max_vocab_count" before loading the dataset, with
      the following exception: With `max_vocab_count`, all the remaining items
      are grouped in a special Out-of-vocabulary item. With `max_num_items`,
      this is not the case. Default: -1.
    categorical_set_split_min_item_frequency: For categorical set splits e.g.
      texts. Minimum number of occurrences of an item to be considered.
      Default: 1.
    compute_permutation_variable_importance: If true, compute the permutation
      variable importance of the model at the end of the training using the
      validation dataset. Enabling this feature can increase the training time
      significantly. Default: False.
    dart_dropout: Dropout rate applied when using the DART i.e. when
      forest_extraction=DART. Default: 0.01.
    early_stopping: Early stopping detects the overfitting of the model and
      halts it training using the validation dataset. If not provided directly,
      the validation dataset is extracted from the training dataset (see
      "validation_ratio" parameter): - `NONE`: No early stopping. All the
      num_trees are trained and kept. - `MIN_LOSS_FINAL`: All the num_trees are
      trained. The model is then truncated to minimize the validation loss i.e.
      some of the trees are discarded as to minimum the validation loss. -
      `LOSS_INCREASE`: Classical early stopping. Stop the training when the
      validation does not decrease for `early_stopping_num_trees_look_ahead`
      trees. Default: "LOSS_INCREASE".
    early_stopping_initial_iteration: 0-based index of the first iteration
      considered for early stopping computation. Increasing this value prevents
      too early stopping due to noisy initial iterations of the learner.
      Default: 10.
    early_stopping_num_trees_look_ahead: Rolling number of trees used to detect
      validation loss increase and trigger early stopping. Default: 30.
    focal_loss_alpha: EXPERIMENTAL. Weighting parameter for focal loss, positive
      samples weighted by alpha, negative samples by (1-alpha). The default 0.5
      value means no active class-level weighting. Only used with focal loss
      i.e. `loss="BINARY_FOCAL_LOSS"` Default: 0.5.
    focal_loss_gamma: EXPERIMENTAL. Exponent of the misprediction exponent term
      in focal loss, corresponds to gamma parameter in
      https://arxiv.org/pdf/1708.02002.pdf. Only used with focal loss i.e.
        `loss="BINARY_FOCAL_LOSS"` Default: 2.0.
    forest_extraction: How to construct the forest: - MART: For Multiple
      Additive Regression Trees. The "classical" way to build a GBDT i.e. each
      tree tries to "correct" the mistakes of the previous trees. - DART: For
      Dropout Additive Regression Trees. A modification of MART proposed in
      http://proceedings.mlr.press/v38/korlakaivinayak15.pdf. Here, each tree
        tries to "correct" the mistakes of a random subset of the previous
        trees.
      Default: "MART".
    goss_alpha: Alpha parameter for the GOSS (Gradient-based One-Side Sampling;
      "See LightGBM: A Highly Efficient Gradient Boosting Decision Tree")
      sampling method. Default: 0.2.
    goss_beta: Beta parameter for the GOSS (Gradient-based One-Side Sampling)
      sampling method. Default: 0.1.
    growing_strategy: How to grow the tree. - `LOCAL`: Each node is split
      independently of the other nodes. In other words, as long as a node
      satisfy the splits "constraints (e.g. maximum depth, minimum number of
      observations), the node will be split. This is the "classical" way to grow
      decision trees. - `BEST_FIRST_GLOBAL`: The node with the best loss
      reduction among all the nodes of the tree is selected for splitting. This
      method is also called "best first" or "leaf-wise growth". See "Best-first
      decision tree learning", Shi and "Additive logistic regression : A
      statistical view of boosting", Friedman for more details. Default:
      "LOCAL".
    honest: In honest trees, different training examples are used to infer the
      structure and the leaf values. This regularization technique trades
      examples for bias estimates. It might increase or reduce the quality of
      the model. See "Generalized Random Forests", Athey et al. In this paper,
      Honest trees are trained with the Random Forest algorithm with a sampling
      without replacement. Default: False.
    honest_fixed_separation: For honest trees only i.e. honest=true. If true, a
      new random separation is generated for each tree. If false, the same
      separation is used for all the trees (e.g., in Gradient Boosted Trees
      containing multiple trees). Default: False.
    honest_ratio_leaf_examples: For honest trees only i.e. honest=true. Ratio of
      examples used to set the leaf values. Default: 0.5.
    in_split_min_examples_check: Whether to check the `min_examples` constraint
      in the split search (i.e. splits leading to one child having less than
      `min_examples` examples are considered invalid) or before the split search
      (i.e. a node can be derived only if it contains more than `min_examples`
      examples). If false, there can be nodes with less than `min_examples`
      training examples. Default: True.
    keep_non_leaf_label_distribution: Whether to keep the node value (i.e. the
      distribution of the labels of the training examples) of non-leaf nodes.
      This information is not used during serving, however it can be used for
      model interpretation as well as hyper parameter tuning. This can take lots
      of space, sometimes accounting for half of the model size. Default: True.
    l1_regularization: L1 regularization applied to the training loss. Impact
      the tree structures and lead values. Default: 0.0.
    l2_categorical_regularization: L2 regularization applied to the training
      loss for categorical features. Impact the tree structures and lead values.
      Default: 1.0.
    l2_regularization: L2 regularization applied to the training loss for all
      features except the categorical ones. Default: 0.0.
    lambda_loss: Lambda regularization applied to certain training loss
      functions. Only for NDCG loss. Default: 1.0.
    loss: The loss optimized by the model. If not specified (DEFAULT) the loss
      is selected automatically according to the \\"task\\" and label
      statistics. For example, if task=CLASSIFICATION and the label has two
      possible values, the loss will be set to BINOMIAL_LOG_LIKELIHOOD. Possible
      values are: - `DEFAULT`: Select the loss automatically according to the
      task and label statistics. - `BINOMIAL_LOG_LIKELIHOOD`: Binomial log
      likelihood. Only valid for binary classification. - `SQUARED_ERROR`: Least
      square loss. Only valid for regression. - `POISSON`: Poisson log
      likelihood loss. Mainly used for counting problems. Only valid for
      regression. - `MULTINOMIAL_LOG_LIKELIHOOD`: Multinomial log likelihood
      i.e. cross-entropy. Only valid for binary or multi-class classification. -
      `LAMBDA_MART_NDCG5`: LambdaMART with NDCG5. - `XE_NDCG_MART`:  Cross
      Entropy Loss NDCG. See arxiv.org/abs/1911.09798. - `BINARY_FOCAL_LOSS`:
      Focal loss. Only valid for binary classification. See
      https://arxiv.org/pdf/1708.02002.pdf. - `POISSON`: Poisson log likelihood.
        Only valid for regression. - `MEAN_AVERAGE_ERROR`: Mean average error
        a.k.a. MAE. For custom losses, pass the loss object here. Note that when
        using custom losses, the link function is deactivated (aka
        apply_link_function is always False).
        Default: "DEFAULT".
    max_depth: Maximum depth of the tree. `max_depth=1` means that all trees
      will be roots. `max_depth=-1` means that tree depth is not restricted by
      this parameter. Values <= -2 will be ignored. Default: 6.
    max_num_nodes: Maximum number of nodes in the tree. Set to -1 to disable
      this limit. Only available for `growing_strategy=BEST_FIRST_GLOBAL`.
      Default: None.
    maximum_model_size_in_memory_in_bytes: Limit the size of the model when
      stored in ram. Different algorithms can enforce this limit differently.
      Note that when models are compiled into an inference, the size of the
      inference engine is generally much smaller than the original model.
      Default: -1.0.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    mhld_oblique_max_num_attributes: For MHLD oblique splits i.e.
      `split_axis=MHLD_OBLIQUE`. Maximum number of attributes in the projection.
      Increasing this value increases the training time. Decreasing this value
      acts as a regularization. The value should be in [2,
      num_numerical_features]. If the value is above the total number of
      numerical features, the value is capped automatically. The value 1 is
      allowed but results in ordinary (non-oblique) splits. Default: None.
    mhld_oblique_sample_attributes: For MHLD oblique splits i.e.
      `split_axis=MHLD_OBLIQUE`. If true, applies the attribute sampling
      controlled by the "num_candidate_attributes" or
      "num_candidate_attributes_ratio" parameters. If false, all the attributes
      are tested. Default: None.
    min_examples: Minimum number of examples in a node. Default: 5.
    missing_value_policy: Method used to handle missing attribute values. -
      `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean
      (in case of numerical attribute) or the most-frequent-item (in case of
      categorical attribute) computed on the entire dataset (i.e. the
      information contained in the data spec). - `LOCAL_IMPUTATION`: Missing
      attribute values are imputed with the mean (numerical attribute) or
      most-frequent-item (in the case of categorical attribute) evaluated on the
      training examples in the current node. - `RANDOM_LOCAL_IMPUTATION`:
      Missing attribute values are imputed from randomly sampled values from the
      training examples in the current node. This method was proposed by Clinic
      et al. in "Random Survival Forests"
      (https://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908043).
        Default: "GLOBAL_IMPUTATION".
    num_candidate_attributes: Number of unique valid attributes tested for each
      node. An attribute is valid if it has at least a valid split. If
      `num_candidate_attributes=0`, the value is set to the classical default
      value for Random Forest: `sqrt(number of input attributes)` in case of
      classification and `number_of_input_attributes / 3` in case of regression.
      If `num_candidate_attributes=-1`, all the attributes are tested. Default:
      -1.
    num_candidate_attributes_ratio: Ratio of attributes tested at each node. If
      set, it is equivalent to `num_candidate_attributes =
      number_of_input_features x num_candidate_attributes_ratio`. The possible
      values are between ]0, and 1] as well as -1. If not set or equal to -1,
      the `num_candidate_attributes` is used. Default: -1.0.
    num_trees: Maximum number of decision trees. The effective number of trained
      tree can be smaller if early stopping is enabled. Default: 300.
    pure_serving_model: Clear the model from any information that is not
      required for model serving. This includes debugging, model interpretation
      and other meta-data. The size of the serialized model can be reduced
      significatively (50% model size reduction is common). This parameter has
      no impact on the quality, serving speed or RAM usage of model serving.
      Default: False.
    random_seed: Random seed for the training of the model. Learners are
      expected to be deterministic by the random seed. Default: 123456.
    sampling_method: Control the sampling of the datasets used to train
      individual trees. - NONE: No sampling is applied. This is equivalent to
      RANDOM sampling with \\"subsample=1\\". - RANDOM (default): Uniform random
      sampling. Automatically selected if "subsample" is set. - GOSS:
      Gradient-based One-Side Sampling. Automatically selected if "goss_alpha"
      or "goss_beta" is set. - SELGB: Selective Gradient Boosting. Automatically
      selected if "selective_gradient_boosting_ratio" is set. Only valid for
      ranking.
        Default: "RANDOM".
    selective_gradient_boosting_ratio: Ratio of the dataset used to train
      individual tree for the selective Gradient Boosting (Selective Gradient
      Boosting for Effective Learning to Rank; Lucchese et al;
      http://quickrank.isti.cnr.it/selective-data/selective-SIGIR2018.pdf)
        sampling method. Default: 0.01.
    shrinkage: Coefficient applied to each tree prediction. A small value (0.02)
      tends to give more accurate results (assuming enough trees are trained),
      but results in larger models. Analogous to neural network learning rate.
      Default: 0.1.
    sorting_strategy: How are sorted the numerical features in order to find the
      splits - PRESORT: The features are pre-sorted at the start of the
      training. This solution is faster but consumes much more memory than
      IN_NODE. - IN_NODE: The features are sorted just before being used in the
      node. This solution is slow but consumes little amount of memory. .
      Default: "PRESORT".
    sparse_oblique_max_num_projections: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Maximum number of projections (applied after
      the num_projections_exponent). Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Increasing "max_num_projections" increases the training time but not the
      inference time. In late stage model development, if every bit of accuracy
      if important, increase this value. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) does not define this hyperparameter.
      Default: None.
    sparse_oblique_normalization: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before
      applying the sparse oblique projections. - `NONE`: No normalization. -
      `STANDARD_DEVIATION`: Normalize the feature by the estimated standard
      deviation on the entire train dataset. Also known as Z-Score
      normalization. - `MIN_MAX`: Normalize the feature by the range (i.e.
      max-min) estimated on the entire train dataset. Default: None.
    sparse_oblique_num_projections_exponent: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node. Increasing this value very likely improves the
      quality of the model, drastically increases the training time, and doe not
      impact the inference time. Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Therefore, increasing this `num_projections_exponent` and possibly
      `max_num_projections` may improve model quality, but will also
      significantly increase training time. Note that the complexity of
      (classic) Random Forests is roughly proportional to
      `num_projections_exponent=0.5`, since it considers sqrt(num_features) for
      a split. The complexity of (classic) GBDT is roughly proportional to
      `num_projections_exponent=1`, since it considers all features for a split.
      The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020)
      recommends values in [1/4, 2]. Default: None.
    sparse_oblique_projection_density_factor: Density of the projections as an
      exponent of the number of features. Independently for each projection,
      each feature has a probability "projection_density_factor / num_features"
      to be considered in the projection. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) calls this parameter `lambda` and
      recommends values in [1, 5]. Increasing this value increases training and
      inference time (on average). This value is best tuned for each dataset.
      Default: None.
    sparse_oblique_weights: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Possible values: - `BINARY`: The oblique
      weights are sampled in {-1,1} (default). - `CONTINUOUS`: The oblique
      weights are be sampled in [-1,1]. Default: None.
    split_axis: What structure of split to consider for numerical features. -
      `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This
      is the "classical" way to train a tree. Default value. - `SPARSE_OBLIQUE`:
      Sparse oblique splits (i.e. random splits one a small number of features)
      from "Sparse Projection Oblique Random Forests", Tomita et al., 2020. -
      `MHLD_OBLIQUE`: Multi-class Hellinger Linear Discriminant splits from
      "Classification Based on Multivariate Contrast Patterns", Canete-Sifuentes
      et al., 2029 Default: "AXIS_ALIGNED".
    subsample: Ratio of the dataset (sampling without replacement) used to train
      individual trees for the random sampling method. If \\"subsample\\" is set
      and if \\"sampling_method\\" is NOT set or set to \\"NONE\\", then
      \\"sampling_method\\" is implicitly set to \\"RANDOM\\". In other words,
      to enable random subsampling, you only need to set "\\"subsample\\".
      Default: 1.0.
    uplift_min_examples_in_treatment: For uplift models only. Minimum number of
      examples per treatment in a node. Default: 5.
    uplift_split_score: For uplift models only. Splitter score i.e. score
      optimized by the splitters. The scores are introduced in "Decision trees
      for uplift modeling with single and multiple treatments", Rzepakowski et
      al. Notation: `p` probability / average value of the positive outcome, `q`
      probability / average value in the control group. - `KULLBACK_LEIBLER` or
      `KL`: - p log (p/q) - `EUCLIDEAN_DISTANCE` or `ED`: (p-q)^2 -
      `CHI_SQUARED` or `CS`: (p-q)^2/q
        Default: "KULLBACK_LEIBLER".
    use_hessian_gain: Use true, uses a formulation of split gain with a hessian
      term i.e. optimizes the splits to minimize the variance of "gradient /
      hessian. Available for all losses except regression. Default: False.
    validation_interval_in_trees: Evaluate the model on the validation set every
      "validation_interval_in_trees" trees. Increasing this value reduce the
      cost of validation and can impact the early stopping policy (as early
      stopping is only tested during the validation). Default: 1.
    validation_ratio: Fraction of the training dataset used for validation if
      not validation dataset is provided. The validation dataset, whether
      provided directly or extracted from the training dataset, is used to
      compute the validation loss, other validation metrics, and possibly
      trigger early stopping (if enabled). When early stopping is disabled, the
      validation dataset is only used for monitoring and does not influence the
      model directly. If the "validation_ratio" is set to 0, early stopping is
      disabled (i.e., it implies setting early_stopping=NONE). Default: 0.1.
    num_threads: Number of threads used to train the model. Different learning
      algorithms use multi-threading differently and with different degree of
      efficiency. If `None`, `num_threads` will be automatically set to the
      number of processors (up to a maximum of 32; or set to 6 if the number of
      processors is not available). Making `num_threads` significantly larger
      than the number of processors can slow-down the training speed. The
      default value logic might change in the future.
    resume_training: If true, the model training resumes from the checkpoint
      stored in the `working_dir` directory. If `working_dir` does not contain
      any model checkpoint, the training starts from the beginning. Resuming
      training is useful in the following situations: (1) The training was
      interrupted by the user (e.g. ctrl+c or "stop" button in a notebook) or
      rescheduled, or (2) the hyper-parameter of the learner was changed e.g.
      increasing the number of trees.
    working_dir: Path to a directory available for the learning algorithm to
      store intermediate computation results. Depending on the learning
      algorithm and parameters, the working_dir might be optional, required, or
      ignored. For instance, distributed training algorithm always need a
      "working_dir", and the gradient boosted tree and hyper-parameter tuners
      will export artefacts to the "working_dir" if provided.
    resume_training_snapshot_interval_seconds: Indicative number of seconds in
      between snapshots when `resume_training=True`. Might be ignored by some
      learners.
    tuner: If set, automatically select the best hyperparameters using the
      provided tuner. When using distributed training, the tuning is
      distributed.
    workers: If set, enable distributed training. "workers" is the list of IP
      addresses of the workers. A worker is a process running
      `ydf.start_worker(port)`.
  """

  def __init__(
      self,
      label: str,
      task: generic_learner.Task = generic_learner.Task.CLASSIFICATION,
      weights: Optional[str] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      features: dataspec.ColumnDefs = None,
      include_all_columns: bool = False,
      max_vocab_count: int = 2000,
      min_vocab_frequency: int = 5,
      discretize_numerical_columns: bool = False,
      num_discretized_numerical_bins: int = 255,
      max_num_scanned_rows_to_infer_semantic: int = 10000,
      max_num_scanned_rows_to_compute_statistics: int = 10000,
      data_spec: Optional[data_spec_pb2.DataSpecification] = None,
      adapt_subsample_for_maximum_training_duration: Optional[bool] = False,
      allow_na_conditions: Optional[bool] = False,
      apply_link_function: Optional[bool] = True,
      categorical_algorithm: Optional[str] = "CART",
      categorical_set_split_greedy_sampling: Optional[float] = 0.1,
      categorical_set_split_max_num_items: Optional[int] = -1,
      categorical_set_split_min_item_frequency: Optional[int] = 1,
      compute_permutation_variable_importance: Optional[bool] = False,
      dart_dropout: Optional[float] = 0.01,
      early_stopping: Optional[str] = "LOSS_INCREASE",
      early_stopping_initial_iteration: Optional[int] = 10,
      early_stopping_num_trees_look_ahead: Optional[int] = 30,
      focal_loss_alpha: Optional[float] = 0.5,
      focal_loss_gamma: Optional[float] = 2.0,
      forest_extraction: Optional[str] = "MART",
      goss_alpha: Optional[float] = 0.2,
      goss_beta: Optional[float] = 0.1,
      growing_strategy: Optional[str] = "LOCAL",
      honest: Optional[bool] = False,
      honest_fixed_separation: Optional[bool] = False,
      honest_ratio_leaf_examples: Optional[float] = 0.5,
      in_split_min_examples_check: Optional[bool] = True,
      keep_non_leaf_label_distribution: Optional[bool] = True,
      l1_regularization: Optional[float] = 0.0,
      l2_categorical_regularization: Optional[float] = 1.0,
      l2_regularization: Optional[float] = 0.0,
      lambda_loss: Optional[float] = 1.0,
      loss: Optional[Union[str, custom_loss.AbstractCustomLoss]] = "DEFAULT",
      max_depth: Optional[int] = 6,
      max_num_nodes: Optional[int] = None,
      maximum_model_size_in_memory_in_bytes: Optional[float] = -1.0,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      mhld_oblique_max_num_attributes: Optional[int] = None,
      mhld_oblique_sample_attributes: Optional[bool] = None,
      min_examples: Optional[int] = 5,
      missing_value_policy: Optional[str] = "GLOBAL_IMPUTATION",
      num_candidate_attributes: Optional[int] = -1,
      num_candidate_attributes_ratio: Optional[float] = -1.0,
      num_trees: Optional[int] = 300,
      pure_serving_model: Optional[bool] = False,
      random_seed: Optional[int] = 123456,
      sampling_method: Optional[str] = "RANDOM",
      selective_gradient_boosting_ratio: Optional[float] = 0.01,
      shrinkage: Optional[float] = 0.1,
      sorting_strategy: Optional[str] = "PRESORT",
      sparse_oblique_max_num_projections: Optional[int] = None,
      sparse_oblique_normalization: Optional[str] = None,
      sparse_oblique_num_projections_exponent: Optional[float] = None,
      sparse_oblique_projection_density_factor: Optional[float] = None,
      sparse_oblique_weights: Optional[str] = None,
      split_axis: Optional[str] = "AXIS_ALIGNED",
      subsample: Optional[float] = 1.0,
      uplift_min_examples_in_treatment: Optional[int] = 5,
      uplift_split_score: Optional[str] = "KULLBACK_LEIBLER",
      use_hessian_gain: Optional[bool] = False,
      validation_interval_in_trees: Optional[int] = 1,
      validation_ratio: Optional[float] = 0.1,
      num_threads: Optional[int] = None,
      working_dir: Optional[str] = None,
      resume_training: bool = False,
      resume_training_snapshot_interval_seconds: int = 1800,
      tuner: Optional[tuner_lib.AbstractTuner] = None,
      workers: Optional[Sequence[str]] = None,
  ):

    hyper_parameters = {
        "adapt_subsample_for_maximum_training_duration": (
            adapt_subsample_for_maximum_training_duration
        ),
        "allow_na_conditions": allow_na_conditions,
        "apply_link_function": apply_link_function,
        "categorical_algorithm": categorical_algorithm,
        "categorical_set_split_greedy_sampling": (
            categorical_set_split_greedy_sampling
        ),
        "categorical_set_split_max_num_items": (
            categorical_set_split_max_num_items
        ),
        "categorical_set_split_min_item_frequency": (
            categorical_set_split_min_item_frequency
        ),
        "compute_permutation_variable_importance": (
            compute_permutation_variable_importance
        ),
        "dart_dropout": dart_dropout,
        "early_stopping": early_stopping,
        "early_stopping_initial_iteration": early_stopping_initial_iteration,
        "early_stopping_num_trees_look_ahead": (
            early_stopping_num_trees_look_ahead
        ),
        "focal_loss_alpha": focal_loss_alpha,
        "focal_loss_gamma": focal_loss_gamma,
        "forest_extraction": forest_extraction,
        "goss_alpha": goss_alpha,
        "goss_beta": goss_beta,
        "growing_strategy": growing_strategy,
        "honest": honest,
        "honest_fixed_separation": honest_fixed_separation,
        "honest_ratio_leaf_examples": honest_ratio_leaf_examples,
        "in_split_min_examples_check": in_split_min_examples_check,
        "keep_non_leaf_label_distribution": keep_non_leaf_label_distribution,
        "l1_regularization": l1_regularization,
        "l2_categorical_regularization": l2_categorical_regularization,
        "l2_regularization": l2_regularization,
        "lambda_loss": lambda_loss,
        "loss": loss,
        "max_depth": max_depth,
        "max_num_nodes": max_num_nodes,
        "maximum_model_size_in_memory_in_bytes": (
            maximum_model_size_in_memory_in_bytes
        ),
        "maximum_training_duration_seconds": maximum_training_duration_seconds,
        "mhld_oblique_max_num_attributes": mhld_oblique_max_num_attributes,
        "mhld_oblique_sample_attributes": mhld_oblique_sample_attributes,
        "min_examples": min_examples,
        "missing_value_policy": missing_value_policy,
        "num_candidate_attributes": num_candidate_attributes,
        "num_candidate_attributes_ratio": num_candidate_attributes_ratio,
        "num_trees": num_trees,
        "pure_serving_model": pure_serving_model,
        "random_seed": random_seed,
        "sampling_method": sampling_method,
        "selective_gradient_boosting_ratio": selective_gradient_boosting_ratio,
        "shrinkage": shrinkage,
        "sorting_strategy": sorting_strategy,
        "sparse_oblique_max_num_projections": (
            sparse_oblique_max_num_projections
        ),
        "sparse_oblique_normalization": sparse_oblique_normalization,
        "sparse_oblique_num_projections_exponent": (
            sparse_oblique_num_projections_exponent
        ),
        "sparse_oblique_projection_density_factor": (
            sparse_oblique_projection_density_factor
        ),
        "sparse_oblique_weights": sparse_oblique_weights,
        "split_axis": split_axis,
        "subsample": subsample,
        "uplift_min_examples_in_treatment": uplift_min_examples_in_treatment,
        "uplift_split_score": uplift_split_score,
        "use_hessian_gain": use_hessian_gain,
        "validation_interval_in_trees": validation_interval_in_trees,
        "validation_ratio": validation_ratio,
    }

    data_spec_args = dataspec.DataSpecInferenceArgs(
        columns=dataspec.normalize_column_defs(features),
        include_all_columns=include_all_columns,
        max_vocab_count=max_vocab_count,
        min_vocab_frequency=min_vocab_frequency,
        discretize_numerical_columns=discretize_numerical_columns,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        max_num_scanned_rows_to_infer_semantic=max_num_scanned_rows_to_infer_semantic,
        max_num_scanned_rows_to_compute_statistics=max_num_scanned_rows_to_compute_statistics,
    )

    deployment_config = self._build_deployment_config(
        num_threads=num_threads,
        resume_training=resume_training,
        resume_training_snapshot_interval_seconds=resume_training_snapshot_interval_seconds,
        working_dir=working_dir,
        workers=workers,
    )

    super().__init__(
        learner_name="GRADIENT_BOOSTED_TREES",
        task=task,
        label=label,
        weights=weights,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        data_spec_args=data_spec_args,
        data_spec=data_spec,
        hyper_parameters=hyper_parameters,
        deployment_config=deployment_config,
        tuner=tuner,
    )

  def train(
      self,
      ds: dataset.InputDataset,
      valid: Optional[dataset.InputDataset] = None,
      verbose: Optional[Union[int, bool]] = None,
  ) -> gradient_boosted_trees_model.GradientBoostedTreesModel:
    """Trains a model on the given dataset.

    Options for dataset reading are given on the learner. Consult the
    documentation of the learner or ydf.create_vertical_dataset() for additional
    information on dataset reading in YDF.

    Usage example:

    ```
    import ydf
    import pandas as pd

    train_ds = pd.read_csv(...)

    learner = ydf.GradientBoostedTreesLearner(label="label")
    model = learner.train(train_ds)
    print(model.summary())
    ```

    If training is interrupted (for example, by interrupting the cell execution
    in Colab), the model will be returned to the state it was in at the moment
    of interruption.

    Args:
      ds: Training dataset.
      valid: Optional validation dataset. Some learners, such as Random Forest,
        do not need validation dataset. Some learners, such as
        GradientBoostedTrees, automatically extract a validation dataset from
        the training dataset if the validation dataset is not provided.
      verbose: Verbose level during training. If None, uses the global verbose
        level of `ydf.verbose`. Levels are: 0 of False: No logs, 1 or True:
        Print a few logs in a notebook; prints all the logs in a terminal. 2:
        Prints all the logs on all surfaces.

    Returns:
      A trained model.
    """
    return super().train(ds=ds, valid=valid, verbose=verbose)

  @classmethod
  def capabilities(cls) -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
        support_max_training_duration=True,
        resume_training=True,
        support_validation_dataset=True,
        support_partial_cache_dataset_format=False,
        support_max_model_size_in_memory=False,
        support_monotonic_constraints=True,
    )

  @classmethod
  def hyperparameter_templates(
      cls,
  ) -> Dict[str, hyperparameters.HyperparameterTemplate]:
    r"""Hyperparameter templates for this Learner.

    Hyperparameter templates are sets of pre-defined hyperparameters for easy
    access to different variants of the learner. Each template is a mapping to a
    set of hyperparameters and can be applied directly on the learner.

    Usage example:
    ```python
    templates = ydf.GradientBoostedTreesLearner.hyperparameter_templates()
    better_defaultv1 = templates["better_defaultv1"]
    # Print a description of the template
    print(better_defaultv1.description)
    # Apply the template's settings on the learner.
    learner = ydf.GradientBoostedTreesLearner(label, **better_defaultv1)
    ```

    Returns:
      Dictionary of the available templates
    """
    return {
        "better_defaultv1": hyperparameters.HyperparameterTemplate(
            name="better_default",
            version=1,
            description=(
                "A configuration that is generally better than the default"
                " parameters without being more expensive."
            ),
            parameters={"growing_strategy": "BEST_FIRST_GLOBAL"},
        ),
        "benchmark_rank1v1": hyperparameters.HyperparameterTemplate(
            name="benchmark_rank1",
            version=1,
            description=(
                "Top ranking hyper-parameters on our benchmark slightly"
                " modified to run in reasonable time."
            ),
            parameters={
                "growing_strategy": "BEST_FIRST_GLOBAL",
                "categorical_algorithm": "RANDOM",
                "split_axis": "SPARSE_OBLIQUE",
                "sparse_oblique_normalization": "MIN_MAX",
                "sparse_oblique_num_projections_exponent": 1.0,
            },
        ),
    }


class DistributedGradientBoostedTreesLearner(generic_learner.GenericLearner):
  r"""Distributed Gradient Boosted Trees learning algorithm.

  Exact distributed version of the Gradient Boosted Tree learning algorithm. See
  the documentation of the non-distributed Gradient Boosted Tree learning
  algorithm for an introduction to GBTs.

  Usage example:

  ```python
  import ydf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")

  model = ydf.DistributedGradientBoostedTreesLearner().train(dataset)

  print(model.summary())
  ```

  Hyperparameters are configured to give reasonable results for typical
  datasets. Hyperparameters can also be modified manually (see descriptions)
  below or by applying the hyperparameter templates available with
  `DistributedGradientBoostedTreesLearner.hyperparameter_templates()` (see this
  function's documentation for
  details).

  Attributes:
    label: Label of the dataset. The label column should not be identified as a
      feature in the `features` parameter.
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT).
    weights: Name of a feature that identifies the weight of each example. If
      weights are not specified, unit weights are assumed. The weight column
      should not be identified as a feature in the `features` parameter.
    ranking_group: Only for `task=Task.RANKING`. Name of a feature that
      identifies queries in a query/document ranking task. The ranking group
      should not be identified as a feature in the `features` parameter.
    uplift_treatment: Only for `task=Task.CATEGORICAL_UPLIFT` and `task=Task`.
      NUMERICAL_UPLIFT. Name of a numerical feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment. Currently, only 0/1 binary treatments are supported.
    features: If None, all columns are used as features. The semantic of the
      features is determined automatically. Otherwise, if
      include_all_columns=False (default) only the column listed in `features`
      are imported. If include_all_columns=True, all the columns are imported as
      features and only the semantic of the columns NOT in `columns` is
      determined automatically. If specified,  defines the order of the features
      - any non-listed features are appended in-order after the specified
      features (if include_all_columns=True). The label, weights, uplift
      treatment and ranking_group columns should not be specified as features.
    include_all_columns: See `features`.
    max_vocab_count: Maximum size of the vocabulary of CATEGORICAL and
      CATEGORICAL_SET columns stored as strings. If more unique values exist,
      only the most frequent values are kept, and the remaining values are
      considered as out-of-vocabulary.
    min_vocab_frequency: Minimum number of occurrence of a value for CATEGORICAL
      and CATEGORICAL_SET columns. Value observed less than
      `min_vocab_frequency` are considered as out-of-vocabulary.
    discretize_numerical_columns: If true, discretize all the numerical columns
      before training. Discretized numerical columns are faster to train with,
      but they can have a negative impact on the model quality. Using
      `discretize_numerical_columns=True` is equivalent as setting the column
      semantic DISCRETIZED_NUMERICAL in the `column` argument. See the
      definition of DISCRETIZED_NUMERICAL for more details.
    num_discretized_numerical_bins: Number of bins used when disretizing
      numerical columns.
    max_num_scanned_rows_to_infer_semantic: Number of rows to scan when
      inferring the column's semantic if it is not explicitly specified. Only
      used when reading from file, in-memory datasets are always read in full.
      Setting this to a lower number will speed up dataset reading, but might
      result in incorrect column semantics. Set to -1 to scan the entire
      dataset.
    max_num_scanned_rows_to_compute_statistics: Number of rows to scan when
      computing a column's statistics. Only used when reading from file,
      in-memory datasets are always read in full. A column's statistics include
      the dictionary for categorical features and the mean / min / max for
      numerical features. Setting this to a lower number will speed up dataset
      reading, but skew statistics in the dataspec, which can hurt model quality
      (e.g. if an important category of a categorical feature is considered
      OOV). Set to -1 to scan the entire dataset.
    data_spec: Dataspec to be used (advanced). If a data spec is given,
      `columns`, `include_all_columns`, `max_vocab_count`,
      `min_vocab_frequency`, `discretize_numerical_columns` and
      `num_discretized_numerical_bins` will be ignored.
    apply_link_function: If true, applies the link function (a.k.a. activation
      function), if any, before returning the model prediction. If false,
      returns the pre-link function model output. For example, in the case of
      binary classification, the pre-link function output is a logic while the
      post-link function is a probability. Default: True.
    force_numerical_discretization: If false, only the numerical column
      safisfying "max_unique_values_for_discretized_numerical" will be
      discretized. If true, all the numerical columns will be discretized.
      Columns with more than "max_unique_values_for_discretized_numerical"
      unique values will be approximated with
      "max_unique_values_for_discretized_numerical" bins. This parameter will
      impact the model training. Default: False.
    max_depth: Maximum depth of the tree. `max_depth=1` means that all trees
      will be roots. `max_depth=-1` means that tree depth is not restricted by
      this parameter. Values <= -2 will be ignored. Default: 6.
    max_unique_values_for_discretized_numerical: Maximum number of unique value
      of a numerical feature to allow its pre-discretization. In case of large
      datasets, discretized numerical features with a small number of unique
      values are more efficient to learn than classical / non-discretized
      numerical features. This parameter does not impact the final model.
      However, it can speed-up or slown the training. Default: 16000.
    maximum_model_size_in_memory_in_bytes: Limit the size of the model when
      stored in ram. Different algorithms can enforce this limit differently.
      Note that when models are compiled into an inference, the size of the
      inference engine is generally much smaller than the original model.
      Default: -1.0.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    min_examples: Minimum number of examples in a node. Default: 5.
    num_candidate_attributes: Number of unique valid attributes tested for each
      node. An attribute is valid if it has at least a valid split. If
      `num_candidate_attributes=0`, the value is set to the classical default
      value for Random Forest: `sqrt(number of input attributes)` in case of
      classification and `number_of_input_attributes / 3` in case of regression.
      If `num_candidate_attributes=-1`, all the attributes are tested. Default:
      -1.
    num_candidate_attributes_ratio: Ratio of attributes tested at each node. If
      set, it is equivalent to `num_candidate_attributes =
      number_of_input_features x num_candidate_attributes_ratio`. The possible
      values are between ]0, and 1] as well as -1. If not set or equal to -1,
      the `num_candidate_attributes` is used. Default: -1.0.
    num_trees: Maximum number of decision trees. The effective number of trained
      tree can be smaller if early stopping is enabled. Default: 300.
    pure_serving_model: Clear the model from any information that is not
      required for model serving. This includes debugging, model interpretation
      and other meta-data. The size of the serialized model can be reduced
      significatively (50% model size reduction is common). This parameter has
      no impact on the quality, serving speed or RAM usage of model serving.
      Default: False.
    random_seed: Random seed for the training of the model. Learners are
      expected to be deterministic by the random seed. Default: 123456.
    shrinkage: Coefficient applied to each tree prediction. A small value (0.02)
      tends to give more accurate results (assuming enough trees are trained),
      but results in larger models. Analogous to neural network learning rate.
      Default: 0.1.
    use_hessian_gain: Use true, uses a formulation of split gain with a hessian
      term i.e. optimizes the splits to minimize the variance of "gradient /
      hessian. Available for all losses except regression. Default: False.
    worker_logs: If true, workers will print training logs. Default: True.
    num_threads: Number of threads used to train the model. Different learning
      algorithms use multi-threading differently and with different degree of
      efficiency. If `None`, `num_threads` will be automatically set to the
      number of processors (up to a maximum of 32; or set to 6 if the number of
      processors is not available). Making `num_threads` significantly larger
      than the number of processors can slow-down the training speed. The
      default value logic might change in the future.
    resume_training: If true, the model training resumes from the checkpoint
      stored in the `working_dir` directory. If `working_dir` does not contain
      any model checkpoint, the training starts from the beginning. Resuming
      training is useful in the following situations: (1) The training was
      interrupted by the user (e.g. ctrl+c or "stop" button in a notebook) or
      rescheduled, or (2) the hyper-parameter of the learner was changed e.g.
      increasing the number of trees.
    working_dir: Path to a directory available for the learning algorithm to
      store intermediate computation results. Depending on the learning
      algorithm and parameters, the working_dir might be optional, required, or
      ignored. For instance, distributed training algorithm always need a
      "working_dir", and the gradient boosted tree and hyper-parameter tuners
      will export artefacts to the "working_dir" if provided.
    resume_training_snapshot_interval_seconds: Indicative number of seconds in
      between snapshots when `resume_training=True`. Might be ignored by some
      learners.
    tuner: If set, automatically select the best hyperparameters using the
      provided tuner. When using distributed training, the tuning is
      distributed.
    workers: If set, enable distributed training. "workers" is the list of IP
      addresses of the workers. A worker is a process running
      `ydf.start_worker(port)`.
  """

  def __init__(
      self,
      label: str,
      task: generic_learner.Task = generic_learner.Task.CLASSIFICATION,
      weights: Optional[str] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      features: dataspec.ColumnDefs = None,
      include_all_columns: bool = False,
      max_vocab_count: int = 2000,
      min_vocab_frequency: int = 5,
      discretize_numerical_columns: bool = False,
      num_discretized_numerical_bins: int = 255,
      max_num_scanned_rows_to_infer_semantic: int = 10000,
      max_num_scanned_rows_to_compute_statistics: int = 10000,
      data_spec: Optional[data_spec_pb2.DataSpecification] = None,
      apply_link_function: Optional[bool] = True,
      force_numerical_discretization: Optional[bool] = False,
      max_depth: Optional[int] = 6,
      max_unique_values_for_discretized_numerical: Optional[int] = 16000,
      maximum_model_size_in_memory_in_bytes: Optional[float] = -1.0,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      min_examples: Optional[int] = 5,
      num_candidate_attributes: Optional[int] = -1,
      num_candidate_attributes_ratio: Optional[float] = -1.0,
      num_trees: Optional[int] = 300,
      pure_serving_model: Optional[bool] = False,
      random_seed: Optional[int] = 123456,
      shrinkage: Optional[float] = 0.1,
      use_hessian_gain: Optional[bool] = False,
      worker_logs: Optional[bool] = True,
      num_threads: Optional[int] = None,
      working_dir: Optional[str] = None,
      resume_training: bool = False,
      resume_training_snapshot_interval_seconds: int = 1800,
      tuner: Optional[tuner_lib.AbstractTuner] = None,
      workers: Optional[Sequence[str]] = None,
  ):

    hyper_parameters = {
        "apply_link_function": apply_link_function,
        "force_numerical_discretization": force_numerical_discretization,
        "max_depth": max_depth,
        "max_unique_values_for_discretized_numerical": (
            max_unique_values_for_discretized_numerical
        ),
        "maximum_model_size_in_memory_in_bytes": (
            maximum_model_size_in_memory_in_bytes
        ),
        "maximum_training_duration_seconds": maximum_training_duration_seconds,
        "min_examples": min_examples,
        "num_candidate_attributes": num_candidate_attributes,
        "num_candidate_attributes_ratio": num_candidate_attributes_ratio,
        "num_trees": num_trees,
        "pure_serving_model": pure_serving_model,
        "random_seed": random_seed,
        "shrinkage": shrinkage,
        "use_hessian_gain": use_hessian_gain,
        "worker_logs": worker_logs,
    }

    data_spec_args = dataspec.DataSpecInferenceArgs(
        columns=dataspec.normalize_column_defs(features),
        include_all_columns=include_all_columns,
        max_vocab_count=max_vocab_count,
        min_vocab_frequency=min_vocab_frequency,
        discretize_numerical_columns=discretize_numerical_columns,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        max_num_scanned_rows_to_infer_semantic=max_num_scanned_rows_to_infer_semantic,
        max_num_scanned_rows_to_compute_statistics=max_num_scanned_rows_to_compute_statistics,
    )

    deployment_config = self._build_deployment_config(
        num_threads=num_threads,
        resume_training=resume_training,
        resume_training_snapshot_interval_seconds=resume_training_snapshot_interval_seconds,
        working_dir=working_dir,
        workers=workers,
    )

    super().__init__(
        learner_name="DISTRIBUTED_GRADIENT_BOOSTED_TREES",
        task=task,
        label=label,
        weights=weights,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        data_spec_args=data_spec_args,
        data_spec=data_spec,
        hyper_parameters=hyper_parameters,
        deployment_config=deployment_config,
        tuner=tuner,
    )

  def train(
      self,
      ds: dataset.InputDataset,
      valid: Optional[dataset.InputDataset] = None,
      verbose: Optional[Union[int, bool]] = None,
  ) -> gradient_boosted_trees_model.GradientBoostedTreesModel:
    """Trains a model on the given dataset.

    Options for dataset reading are given on the learner. Consult the
    documentation of the learner or ydf.create_vertical_dataset() for additional
    information on dataset reading in YDF.

    Usage example:

    ```
    import ydf
    import pandas as pd

    train_ds = pd.read_csv(...)

    learner = ydf.DistributedGradientBoostedTreesLearner(label="label")
    model = learner.train(train_ds)
    print(model.summary())
    ```

    If training is interrupted (for example, by interrupting the cell execution
    in Colab), the model will be returned to the state it was in at the moment
    of interruption.

    Args:
      ds: Training dataset.
      valid: Optional validation dataset. Some learners, such as Random Forest,
        do not need validation dataset. Some learners, such as
        GradientBoostedTrees, automatically extract a validation dataset from
        the training dataset if the validation dataset is not provided.
      verbose: Verbose level during training. If None, uses the global verbose
        level of `ydf.verbose`. Levels are: 0 of False: No logs, 1 or True:
        Print a few logs in a notebook; prints all the logs in a terminal. 2:
        Prints all the logs on all surfaces.

    Returns:
      A trained model.
    """
    return super().train(ds=ds, valid=valid, verbose=verbose)

  @classmethod
  def capabilities(cls) -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
        support_max_training_duration=False,
        resume_training=True,
        support_validation_dataset=False,
        support_partial_cache_dataset_format=True,
        support_max_model_size_in_memory=False,
        support_monotonic_constraints=False,
    )

  @classmethod
  def hyperparameter_templates(
      cls,
  ) -> Dict[str, hyperparameters.HyperparameterTemplate]:
    r"""Hyperparameter templates for this Learner.

    This learner currently does not provide any hyperparameter templates, this
    method is provided for consistency with other learners.

    Returns:
      Empty dictionary.
    """
    return {}


class CartLearner(generic_learner.GenericLearner):
  r"""Cart learning algorithm.

  A CART (Classification and Regression Trees) a decision tree. The non-leaf
  nodes contains conditions (also known as splits) while the leaf nodes contain
  prediction values. The training dataset is divided in two parts. The first is
  used to grow the tree while the second is used to prune the tree.

  Usage example:

  ```python
  import ydf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")

  model = ydf.CartLearner().train(dataset)

  print(model.summary())
  ```

  Hyperparameters are configured to give reasonable results for typical
  datasets. Hyperparameters can also be modified manually (see descriptions)
  below or by applying the hyperparameter templates available with
  `CartLearner.hyperparameter_templates()` (see this function's documentation
  for
  details).

  Attributes:
    label: Label of the dataset. The label column should not be identified as a
      feature in the `features` parameter.
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT).
    weights: Name of a feature that identifies the weight of each example. If
      weights are not specified, unit weights are assumed. The weight column
      should not be identified as a feature in the `features` parameter.
    ranking_group: Only for `task=Task.RANKING`. Name of a feature that
      identifies queries in a query/document ranking task. The ranking group
      should not be identified as a feature in the `features` parameter.
    uplift_treatment: Only for `task=Task.CATEGORICAL_UPLIFT` and `task=Task`.
      NUMERICAL_UPLIFT. Name of a numerical feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment. Currently, only 0/1 binary treatments are supported.
    features: If None, all columns are used as features. The semantic of the
      features is determined automatically. Otherwise, if
      include_all_columns=False (default) only the column listed in `features`
      are imported. If include_all_columns=True, all the columns are imported as
      features and only the semantic of the columns NOT in `columns` is
      determined automatically. If specified,  defines the order of the features
      - any non-listed features are appended in-order after the specified
      features (if include_all_columns=True). The label, weights, uplift
      treatment and ranking_group columns should not be specified as features.
    include_all_columns: See `features`.
    max_vocab_count: Maximum size of the vocabulary of CATEGORICAL and
      CATEGORICAL_SET columns stored as strings. If more unique values exist,
      only the most frequent values are kept, and the remaining values are
      considered as out-of-vocabulary.
    min_vocab_frequency: Minimum number of occurrence of a value for CATEGORICAL
      and CATEGORICAL_SET columns. Value observed less than
      `min_vocab_frequency` are considered as out-of-vocabulary.
    discretize_numerical_columns: If true, discretize all the numerical columns
      before training. Discretized numerical columns are faster to train with,
      but they can have a negative impact on the model quality. Using
      `discretize_numerical_columns=True` is equivalent as setting the column
      semantic DISCRETIZED_NUMERICAL in the `column` argument. See the
      definition of DISCRETIZED_NUMERICAL for more details.
    num_discretized_numerical_bins: Number of bins used when disretizing
      numerical columns.
    max_num_scanned_rows_to_infer_semantic: Number of rows to scan when
      inferring the column's semantic if it is not explicitly specified. Only
      used when reading from file, in-memory datasets are always read in full.
      Setting this to a lower number will speed up dataset reading, but might
      result in incorrect column semantics. Set to -1 to scan the entire
      dataset.
    max_num_scanned_rows_to_compute_statistics: Number of rows to scan when
      computing a column's statistics. Only used when reading from file,
      in-memory datasets are always read in full. A column's statistics include
      the dictionary for categorical features and the mean / min / max for
      numerical features. Setting this to a lower number will speed up dataset
      reading, but skew statistics in the dataspec, which can hurt model quality
      (e.g. if an important category of a categorical feature is considered
      OOV). Set to -1 to scan the entire dataset.
    data_spec: Dataspec to be used (advanced). If a data spec is given,
      `columns`, `include_all_columns`, `max_vocab_count`,
      `min_vocab_frequency`, `discretize_numerical_columns` and
      `num_discretized_numerical_bins` will be ignored.
    allow_na_conditions: If true, the tree training evaluates conditions of the
      type `X is NA` i.e. `X is missing`. Default: False.
    categorical_algorithm: How to learn splits on categorical attributes. -
      `CART`: CART algorithm. Find categorical splits of the form "value \\in
      mask". The solution is exact for binary classification, regression and
      ranking. It is approximated for multi-class classification. This is a good
      first algorithm to use. In case of overfitting (very small dataset, large
      dictionary), the "random" algorithm is a good alternative. - `ONE_HOT`:
      One-hot encoding. Find the optimal categorical split of the form
      "attribute == param". This method is similar (but more efficient) than
      converting converting each possible categorical value into a boolean
      feature. This method is available for comparison purpose and generally
      performs worse than other alternatives. - `RANDOM`: Best splits among a
      set of random candidate. Find the a categorical split of the form "value
      \\in mask" using a random search. This solution can be seen as an
      approximation of the CART algorithm. This method is a strong alternative
      to CART. This algorithm is inspired from section "5.1 Categorical
      Variables" of "Random Forest", 2001.
        Default: "CART".
    categorical_set_split_greedy_sampling: For categorical set splits e.g.
      texts. Probability for a categorical value to be a candidate for the
      positive set. The sampling is applied once per node (i.e. not at every
      step of the greedy optimization). Default: 0.1.
    categorical_set_split_max_num_items: For categorical set splits e.g. texts.
      Maximum number of items (prior to the sampling). If more items are
      available, the least frequent items are ignored. Changing this value is
      similar to change the "max_vocab_count" before loading the dataset, with
      the following exception: With `max_vocab_count`, all the remaining items
      are grouped in a special Out-of-vocabulary item. With `max_num_items`,
      this is not the case. Default: -1.
    categorical_set_split_min_item_frequency: For categorical set splits e.g.
      texts. Minimum number of occurrences of an item to be considered.
      Default: 1.
    growing_strategy: How to grow the tree. - `LOCAL`: Each node is split
      independently of the other nodes. In other words, as long as a node
      satisfy the splits "constraints (e.g. maximum depth, minimum number of
      observations), the node will be split. This is the "classical" way to grow
      decision trees. - `BEST_FIRST_GLOBAL`: The node with the best loss
      reduction among all the nodes of the tree is selected for splitting. This
      method is also called "best first" or "leaf-wise growth". See "Best-first
      decision tree learning", Shi and "Additive logistic regression : A
      statistical view of boosting", Friedman for more details. Default:
      "LOCAL".
    honest: In honest trees, different training examples are used to infer the
      structure and the leaf values. This regularization technique trades
      examples for bias estimates. It might increase or reduce the quality of
      the model. See "Generalized Random Forests", Athey et al. In this paper,
      Honest trees are trained with the Random Forest algorithm with a sampling
      without replacement. Default: False.
    honest_fixed_separation: For honest trees only i.e. honest=true. If true, a
      new random separation is generated for each tree. If false, the same
      separation is used for all the trees (e.g., in Gradient Boosted Trees
      containing multiple trees). Default: False.
    honest_ratio_leaf_examples: For honest trees only i.e. honest=true. Ratio of
      examples used to set the leaf values. Default: 0.5.
    in_split_min_examples_check: Whether to check the `min_examples` constraint
      in the split search (i.e. splits leading to one child having less than
      `min_examples` examples are considered invalid) or before the split search
      (i.e. a node can be derived only if it contains more than `min_examples`
      examples). If false, there can be nodes with less than `min_examples`
      training examples. Default: True.
    keep_non_leaf_label_distribution: Whether to keep the node value (i.e. the
      distribution of the labels of the training examples) of non-leaf nodes.
      This information is not used during serving, however it can be used for
      model interpretation as well as hyper parameter tuning. This can take lots
      of space, sometimes accounting for half of the model size. Default: True.
    max_depth: Maximum depth of the tree. `max_depth=1` means that all trees
      will be roots. `max_depth=-1` means that tree depth is not restricted by
      this parameter. Values <= -2 will be ignored. Default: 16.
    max_num_nodes: Maximum number of nodes in the tree. Set to -1 to disable
      this limit. Only available for `growing_strategy=BEST_FIRST_GLOBAL`.
      Default: None.
    maximum_model_size_in_memory_in_bytes: Limit the size of the model when
      stored in ram. Different algorithms can enforce this limit differently.
      Note that when models are compiled into an inference, the size of the
      inference engine is generally much smaller than the original model.
      Default: -1.0.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    mhld_oblique_max_num_attributes: For MHLD oblique splits i.e.
      `split_axis=MHLD_OBLIQUE`. Maximum number of attributes in the projection.
      Increasing this value increases the training time. Decreasing this value
      acts as a regularization. The value should be in [2,
      num_numerical_features]. If the value is above the total number of
      numerical features, the value is capped automatically. The value 1 is
      allowed but results in ordinary (non-oblique) splits. Default: None.
    mhld_oblique_sample_attributes: For MHLD oblique splits i.e.
      `split_axis=MHLD_OBLIQUE`. If true, applies the attribute sampling
      controlled by the "num_candidate_attributes" or
      "num_candidate_attributes_ratio" parameters. If false, all the attributes
      are tested. Default: None.
    min_examples: Minimum number of examples in a node. Default: 5.
    missing_value_policy: Method used to handle missing attribute values. -
      `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean
      (in case of numerical attribute) or the most-frequent-item (in case of
      categorical attribute) computed on the entire dataset (i.e. the
      information contained in the data spec). - `LOCAL_IMPUTATION`: Missing
      attribute values are imputed with the mean (numerical attribute) or
      most-frequent-item (in the case of categorical attribute) evaluated on the
      training examples in the current node. - `RANDOM_LOCAL_IMPUTATION`:
      Missing attribute values are imputed from randomly sampled values from the
      training examples in the current node. This method was proposed by Clinic
      et al. in "Random Survival Forests"
      (https://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908043).
        Default: "GLOBAL_IMPUTATION".
    num_candidate_attributes: Number of unique valid attributes tested for each
      node. An attribute is valid if it has at least a valid split. If
      `num_candidate_attributes=0`, the value is set to the classical default
      value for Random Forest: `sqrt(number of input attributes)` in case of
      classification and `number_of_input_attributes / 3` in case of regression.
      If `num_candidate_attributes=-1`, all the attributes are tested. Default:
      0.
    num_candidate_attributes_ratio: Ratio of attributes tested at each node. If
      set, it is equivalent to `num_candidate_attributes =
      number_of_input_features x num_candidate_attributes_ratio`. The possible
      values are between ]0, and 1] as well as -1. If not set or equal to -1,
      the `num_candidate_attributes` is used. Default: -1.0.
    pure_serving_model: Clear the model from any information that is not
      required for model serving. This includes debugging, model interpretation
      and other meta-data. The size of the serialized model can be reduced
      significatively (50% model size reduction is common). This parameter has
      no impact on the quality, serving speed or RAM usage of model serving.
      Default: False.
    random_seed: Random seed for the training of the model. Learners are
      expected to be deterministic by the random seed. Default: 123456.
    sorting_strategy: How are sorted the numerical features in order to find the
      splits - PRESORT: The features are pre-sorted at the start of the
      training. This solution is faster but consumes much more memory than
      IN_NODE. - IN_NODE: The features are sorted just before being used in the
      node. This solution is slow but consumes little amount of memory. .
      Default: "PRESORT".
    sparse_oblique_max_num_projections: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Maximum number of projections (applied after
      the num_projections_exponent). Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Increasing "max_num_projections" increases the training time but not the
      inference time. In late stage model development, if every bit of accuracy
      if important, increase this value. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) does not define this hyperparameter.
      Default: None.
    sparse_oblique_normalization: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before
      applying the sparse oblique projections. - `NONE`: No normalization. -
      `STANDARD_DEVIATION`: Normalize the feature by the estimated standard
      deviation on the entire train dataset. Also known as Z-Score
      normalization. - `MIN_MAX`: Normalize the feature by the range (i.e.
      max-min) estimated on the entire train dataset. Default: None.
    sparse_oblique_num_projections_exponent: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node. Increasing this value very likely improves the
      quality of the model, drastically increases the training time, and doe not
      impact the inference time. Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Therefore, increasing this `num_projections_exponent` and possibly
      `max_num_projections` may improve model quality, but will also
      significantly increase training time. Note that the complexity of
      (classic) Random Forests is roughly proportional to
      `num_projections_exponent=0.5`, since it considers sqrt(num_features) for
      a split. The complexity of (classic) GBDT is roughly proportional to
      `num_projections_exponent=1`, since it considers all features for a split.
      The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020)
      recommends values in [1/4, 2]. Default: None.
    sparse_oblique_projection_density_factor: Density of the projections as an
      exponent of the number of features. Independently for each projection,
      each feature has a probability "projection_density_factor / num_features"
      to be considered in the projection. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) calls this parameter `lambda` and
      recommends values in [1, 5]. Increasing this value increases training and
      inference time (on average). This value is best tuned for each dataset.
      Default: None.
    sparse_oblique_weights: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Possible values: - `BINARY`: The oblique
      weights are sampled in {-1,1} (default). - `CONTINUOUS`: The oblique
      weights are be sampled in [-1,1]. Default: None.
    split_axis: What structure of split to consider for numerical features. -
      `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This
      is the "classical" way to train a tree. Default value. - `SPARSE_OBLIQUE`:
      Sparse oblique splits (i.e. random splits one a small number of features)
      from "Sparse Projection Oblique Random Forests", Tomita et al., 2020. -
      `MHLD_OBLIQUE`: Multi-class Hellinger Linear Discriminant splits from
      "Classification Based on Multivariate Contrast Patterns", Canete-Sifuentes
      et al., 2029 Default: "AXIS_ALIGNED".
    uplift_min_examples_in_treatment: For uplift models only. Minimum number of
      examples per treatment in a node. Default: 5.
    uplift_split_score: For uplift models only. Splitter score i.e. score
      optimized by the splitters. The scores are introduced in "Decision trees
      for uplift modeling with single and multiple treatments", Rzepakowski et
      al. Notation: `p` probability / average value of the positive outcome, `q`
      probability / average value in the control group. - `KULLBACK_LEIBLER` or
      `KL`: - p log (p/q) - `EUCLIDEAN_DISTANCE` or `ED`: (p-q)^2 -
      `CHI_SQUARED` or `CS`: (p-q)^2/q
        Default: "KULLBACK_LEIBLER".
    validation_ratio: Ratio of the training dataset used to create the
      validation dataset for pruning the tree. If set to 0, the entire dataset
      is used for training, and the tree is not pruned. Default: 0.1.
    num_threads: Number of threads used to train the model. Different learning
      algorithms use multi-threading differently and with different degree of
      efficiency. If `None`, `num_threads` will be automatically set to the
      number of processors (up to a maximum of 32; or set to 6 if the number of
      processors is not available). Making `num_threads` significantly larger
      than the number of processors can slow-down the training speed. The
      default value logic might change in the future.
    resume_training: If true, the model training resumes from the checkpoint
      stored in the `working_dir` directory. If `working_dir` does not contain
      any model checkpoint, the training starts from the beginning. Resuming
      training is useful in the following situations: (1) The training was
      interrupted by the user (e.g. ctrl+c or "stop" button in a notebook) or
      rescheduled, or (2) the hyper-parameter of the learner was changed e.g.
      increasing the number of trees.
    working_dir: Path to a directory available for the learning algorithm to
      store intermediate computation results. Depending on the learning
      algorithm and parameters, the working_dir might be optional, required, or
      ignored. For instance, distributed training algorithm always need a
      "working_dir", and the gradient boosted tree and hyper-parameter tuners
      will export artefacts to the "working_dir" if provided.
    resume_training_snapshot_interval_seconds: Indicative number of seconds in
      between snapshots when `resume_training=True`. Might be ignored by some
      learners.
    tuner: If set, automatically select the best hyperparameters using the
      provided tuner. When using distributed training, the tuning is
      distributed.
    workers: If set, enable distributed training. "workers" is the list of IP
      addresses of the workers. A worker is a process running
      `ydf.start_worker(port)`.
  """

  def __init__(
      self,
      label: str,
      task: generic_learner.Task = generic_learner.Task.CLASSIFICATION,
      weights: Optional[str] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      features: dataspec.ColumnDefs = None,
      include_all_columns: bool = False,
      max_vocab_count: int = 2000,
      min_vocab_frequency: int = 5,
      discretize_numerical_columns: bool = False,
      num_discretized_numerical_bins: int = 255,
      max_num_scanned_rows_to_infer_semantic: int = 10000,
      max_num_scanned_rows_to_compute_statistics: int = 10000,
      data_spec: Optional[data_spec_pb2.DataSpecification] = None,
      allow_na_conditions: Optional[bool] = False,
      categorical_algorithm: Optional[str] = "CART",
      categorical_set_split_greedy_sampling: Optional[float] = 0.1,
      categorical_set_split_max_num_items: Optional[int] = -1,
      categorical_set_split_min_item_frequency: Optional[int] = 1,
      growing_strategy: Optional[str] = "LOCAL",
      honest: Optional[bool] = False,
      honest_fixed_separation: Optional[bool] = False,
      honest_ratio_leaf_examples: Optional[float] = 0.5,
      in_split_min_examples_check: Optional[bool] = True,
      keep_non_leaf_label_distribution: Optional[bool] = True,
      max_depth: Optional[int] = 16,
      max_num_nodes: Optional[int] = None,
      maximum_model_size_in_memory_in_bytes: Optional[float] = -1.0,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      mhld_oblique_max_num_attributes: Optional[int] = None,
      mhld_oblique_sample_attributes: Optional[bool] = None,
      min_examples: Optional[int] = 5,
      missing_value_policy: Optional[str] = "GLOBAL_IMPUTATION",
      num_candidate_attributes: Optional[int] = 0,
      num_candidate_attributes_ratio: Optional[float] = -1.0,
      pure_serving_model: Optional[bool] = False,
      random_seed: Optional[int] = 123456,
      sorting_strategy: Optional[str] = "PRESORT",
      sparse_oblique_max_num_projections: Optional[int] = None,
      sparse_oblique_normalization: Optional[str] = None,
      sparse_oblique_num_projections_exponent: Optional[float] = None,
      sparse_oblique_projection_density_factor: Optional[float] = None,
      sparse_oblique_weights: Optional[str] = None,
      split_axis: Optional[str] = "AXIS_ALIGNED",
      uplift_min_examples_in_treatment: Optional[int] = 5,
      uplift_split_score: Optional[str] = "KULLBACK_LEIBLER",
      validation_ratio: Optional[float] = 0.1,
      num_threads: Optional[int] = None,
      working_dir: Optional[str] = None,
      resume_training: bool = False,
      resume_training_snapshot_interval_seconds: int = 1800,
      tuner: Optional[tuner_lib.AbstractTuner] = None,
      workers: Optional[Sequence[str]] = None,
  ):

    hyper_parameters = {
        "allow_na_conditions": allow_na_conditions,
        "categorical_algorithm": categorical_algorithm,
        "categorical_set_split_greedy_sampling": (
            categorical_set_split_greedy_sampling
        ),
        "categorical_set_split_max_num_items": (
            categorical_set_split_max_num_items
        ),
        "categorical_set_split_min_item_frequency": (
            categorical_set_split_min_item_frequency
        ),
        "growing_strategy": growing_strategy,
        "honest": honest,
        "honest_fixed_separation": honest_fixed_separation,
        "honest_ratio_leaf_examples": honest_ratio_leaf_examples,
        "in_split_min_examples_check": in_split_min_examples_check,
        "keep_non_leaf_label_distribution": keep_non_leaf_label_distribution,
        "max_depth": max_depth,
        "max_num_nodes": max_num_nodes,
        "maximum_model_size_in_memory_in_bytes": (
            maximum_model_size_in_memory_in_bytes
        ),
        "maximum_training_duration_seconds": maximum_training_duration_seconds,
        "mhld_oblique_max_num_attributes": mhld_oblique_max_num_attributes,
        "mhld_oblique_sample_attributes": mhld_oblique_sample_attributes,
        "min_examples": min_examples,
        "missing_value_policy": missing_value_policy,
        "num_candidate_attributes": num_candidate_attributes,
        "num_candidate_attributes_ratio": num_candidate_attributes_ratio,
        "pure_serving_model": pure_serving_model,
        "random_seed": random_seed,
        "sorting_strategy": sorting_strategy,
        "sparse_oblique_max_num_projections": (
            sparse_oblique_max_num_projections
        ),
        "sparse_oblique_normalization": sparse_oblique_normalization,
        "sparse_oblique_num_projections_exponent": (
            sparse_oblique_num_projections_exponent
        ),
        "sparse_oblique_projection_density_factor": (
            sparse_oblique_projection_density_factor
        ),
        "sparse_oblique_weights": sparse_oblique_weights,
        "split_axis": split_axis,
        "uplift_min_examples_in_treatment": uplift_min_examples_in_treatment,
        "uplift_split_score": uplift_split_score,
        "validation_ratio": validation_ratio,
    }

    data_spec_args = dataspec.DataSpecInferenceArgs(
        columns=dataspec.normalize_column_defs(features),
        include_all_columns=include_all_columns,
        max_vocab_count=max_vocab_count,
        min_vocab_frequency=min_vocab_frequency,
        discretize_numerical_columns=discretize_numerical_columns,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        max_num_scanned_rows_to_infer_semantic=max_num_scanned_rows_to_infer_semantic,
        max_num_scanned_rows_to_compute_statistics=max_num_scanned_rows_to_compute_statistics,
    )

    deployment_config = self._build_deployment_config(
        num_threads=num_threads,
        resume_training=resume_training,
        resume_training_snapshot_interval_seconds=resume_training_snapshot_interval_seconds,
        working_dir=working_dir,
        workers=workers,
    )

    super().__init__(
        learner_name="CART",
        task=task,
        label=label,
        weights=weights,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        data_spec_args=data_spec_args,
        data_spec=data_spec,
        hyper_parameters=hyper_parameters,
        deployment_config=deployment_config,
        tuner=tuner,
    )

  def train(
      self,
      ds: dataset.InputDataset,
      valid: Optional[dataset.InputDataset] = None,
      verbose: Optional[Union[int, bool]] = None,
  ) -> random_forest_model.RandomForestModel:
    """Trains a model on the given dataset.

    Options for dataset reading are given on the learner. Consult the
    documentation of the learner or ydf.create_vertical_dataset() for additional
    information on dataset reading in YDF.

    Usage example:

    ```
    import ydf
    import pandas as pd

    train_ds = pd.read_csv(...)

    learner = ydf.CartLearner(label="label")
    model = learner.train(train_ds)
    print(model.summary())
    ```

    If training is interrupted (for example, by interrupting the cell execution
    in Colab), the model will be returned to the state it was in at the moment
    of interruption.

    Args:
      ds: Training dataset.
      valid: Optional validation dataset. Some learners, such as Random Forest,
        do not need validation dataset. Some learners, such as
        GradientBoostedTrees, automatically extract a validation dataset from
        the training dataset if the validation dataset is not provided.
      verbose: Verbose level during training. If None, uses the global verbose
        level of `ydf.verbose`. Levels are: 0 of False: No logs, 1 or True:
        Print a few logs in a notebook; prints all the logs in a terminal. 2:
        Prints all the logs on all surfaces.

    Returns:
      A trained model.
    """
    return super().train(ds=ds, valid=valid, verbose=verbose)

  @classmethod
  def capabilities(cls) -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
        support_max_training_duration=True,
        resume_training=False,
        support_validation_dataset=True,
        support_partial_cache_dataset_format=False,
        support_max_model_size_in_memory=False,
        support_monotonic_constraints=False,
    )

  @classmethod
  def hyperparameter_templates(
      cls,
  ) -> Dict[str, hyperparameters.HyperparameterTemplate]:
    r"""Hyperparameter templates for this Learner.

    This learner currently does not provide any hyperparameter templates, this
    method is provided for consistency with other learners.

    Returns:
      Empty dictionary.
    """
    return {}
