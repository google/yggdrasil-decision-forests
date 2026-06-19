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

"""Unit tests of the tuner."""

from absl.testing import absltest
from absl.testing import parameterized

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.learner.hyperparameters_optimizer import hyperparameters_optimizer_pb2
from yggdrasil_decision_forests.learner.hyperparameters_optimizer.optimizers import random_pb2
from yggdrasil_decision_forests.model import hyperparameter_pb2
from ydf.dataset import dataspec
from ydf.learner import specialized_learners
from ydf.learner import tuner as tuner_lib
from ydf.model import generic_model
from ydf.utils import test_utils
from yggdrasil_decision_forests.utils import fold_generator_pb2

DiscreteCandidates = hyperparameter_pb2.HyperParameterSpace.DiscreteCandidates
Field = hyperparameter_pb2.HyperParameterSpace.Field
Value = hyperparameter_pb2.GenericHyperParameters.Value
HyperParametersOptimizerLearnerTrainingConfig = (
    hyperparameters_optimizer_pb2.HyperParametersOptimizerLearnerTrainingConfig
)


class TunerTest(parameterized.TestCase):

  def test_to_proto(self):
    # Define toy tuner
    tuner = tuner_lib.RandomSearchTuner(
        num_trials=20,
        parallel_trials=2,
        max_trial_duration=10,
    )
    tuner.choice("a", [1, 2, 3])
    tuner.choice("b", [1.0, 2.0, 3.0])
    tuner.choice("c", ["x", "y"])

    s = tuner.choice("c", ["v", "w"], merge=True)
    s.choice("d", [1, 2, 3])

    # Check internal state
    self.assertEqual(tuner.parallel_trials, 2)

    expected_proto = abstract_learner_pb2.TrainingConfig(
        learner="HYPERPARAMETER_OPTIMIZER"
    )

    expected_optimizer = HyperParametersOptimizerLearnerTrainingConfig(
        optimizer=hyperparameters_optimizer_pb2.Optimizer(
            optimizer_key="RANDOM",
            parallel_trials=2,
        ),
        base_learner=abstract_learner_pb2.TrainingConfig(
            maximum_training_duration_seconds=10
        ),
        evaluation=hyperparameters_optimizer_pb2.Evaluation(
            self_model_evaluation=hyperparameters_optimizer_pb2.Evaluation.SelfEvaluation()
        ),
        search_space=hyperparameter_pb2.HyperParameterSpace(
            fields=[
                Field(
                    name="a",
                    discrete_candidates=DiscreteCandidates(
                        possible_values=[
                            Value(integer=1),
                            Value(integer=2),
                            Value(integer=3),
                        ],
                    ),
                ),
                Field(
                    name="b",
                    discrete_candidates=DiscreteCandidates(
                        possible_values=[
                            Value(real=1),
                            Value(real=2),
                            Value(real=3),
                        ]
                    ),
                ),
                Field(
                    name="c",
                    discrete_candidates=DiscreteCandidates(
                        possible_values=[
                            Value(categorical="x"),
                            Value(categorical="y"),
                            Value(categorical="v"),
                            Value(categorical="w"),
                        ]
                    ),
                    children=[
                        Field(
                            name="d",
                            discrete_candidates=DiscreteCandidates(
                                possible_values=[
                                    Value(integer=1),
                                    Value(integer=2),
                                    Value(integer=3),
                                ]
                            ),
                            parent_discrete_values=DiscreteCandidates(
                                possible_values=[
                                    Value(categorical="v"),
                                    Value(categorical="w"),
                                ],
                            ),
                        ),
                    ],
                ),
            ],
        ),
    )

    expected_random_optimizer = random_pb2.RandomOptimizerConfig(num_trials=20)

    # Note: Extension construction is not supported.
    expected_optimizer.optimizer.Extensions[random_pb2.random].CopyFrom(
        expected_random_optimizer
    )
    expected_proto.Extensions[
        hyperparameters_optimizer_pb2.hyperparameters_optimizer_config
    ].CopyFrom(expected_optimizer)

    test_utils.assertProto2Equal(self, tuner.train_config, expected_proto)

  def test_error_no_values(self):
    tuner = tuner_lib.RandomSearchTuner(num_trials=20)
    with self.assertRaisesRegex(ValueError, "The list of values is empty"):
      tuner.choice("a", [])

  def test_merging_does_not_exist(self):
    tuner = tuner_lib.RandomSearchTuner(num_trials=20)
    with self.assertRaisesRegex(
        ValueError, "hyperparameter 'a' does not already exist"
    ):
      tuner.choice("a", [1, 2], merge=True)

  def test_merging_already_exist(self):
    tuner = tuner_lib.RandomSearchTuner(num_trials=20)
    tuner.choice("a", [1, 2])

    with self.assertRaisesRegex(
        ValueError, "The hyperparameter 'a' already exist"
    ):
      tuner.choice("a", [3, 4])

  def test_merging_good(self):
    tuner = tuner_lib.RandomSearchTuner(num_trials=20)
    tuner.choice("a", [1, 2])
    tuner.choice("a", [3, 4], merge=True)

    expected_proto = abstract_learner_pb2.TrainingConfig(
        learner="HYPERPARAMETER_OPTIMIZER"
    )

    expected_optimizer = HyperParametersOptimizerLearnerTrainingConfig(
        optimizer=hyperparameters_optimizer_pb2.Optimizer(
            optimizer_key="RANDOM",
            parallel_trials=1,
        ),
        evaluation=hyperparameters_optimizer_pb2.Evaluation(
            self_model_evaluation=hyperparameters_optimizer_pb2.Evaluation.SelfEvaluation()
        ),
        search_space=hyperparameter_pb2.HyperParameterSpace(
            fields=[
                Field(
                    name="a",
                    discrete_candidates=DiscreteCandidates(
                        possible_values=[
                            Value(integer=1),
                            Value(integer=2),
                            Value(integer=3),
                            Value(integer=4),
                        ],
                    ),
                ),
            ],
        ),
    )
    expected_random_optimizer = random_pb2.RandomOptimizerConfig(num_trials=20)

    # Note: Extension construction is not supported.
    expected_optimizer.optimizer.Extensions[random_pb2.random].CopyFrom(
        expected_random_optimizer
    )
    expected_proto.Extensions[
        hyperparameters_optimizer_pb2.hyperparameters_optimizer_config
    ].CopyFrom(expected_optimizer)

    test_utils.assertProto2Equal(self, tuner.train_config, expected_proto)

  def test_to_proto_cross_validation(self):
    # Define toy tuner
    tuner = tuner_lib.RandomSearchTuner(
        num_trials=20,
        parallel_trials=2,
        max_trial_duration=10,
        cross_validation=True,
        cross_validation_num_folds=2,
    )
    tuner.choice("a", [1, 2, 3])
    tuner.choice("b", [1.0, 2.0, 3.0])
    tuner.choice("c", ["x", "y"])

    s = tuner.choice("c", ["v", "w"], merge=True)
    s.choice("d", [1, 2, 3])

    # Check internal state
    self.assertEqual(tuner.parallel_trials, 2)

    expected_proto = abstract_learner_pb2.TrainingConfig(
        learner="HYPERPARAMETER_OPTIMIZER"
    )

    expected_optimizer = HyperParametersOptimizerLearnerTrainingConfig(
        optimizer=hyperparameters_optimizer_pb2.Optimizer(
            optimizer_key="RANDOM",
            parallel_trials=2,
        ),
        base_learner=abstract_learner_pb2.TrainingConfig(
            maximum_training_duration_seconds=10
        ),
        evaluation=hyperparameters_optimizer_pb2.Evaluation(
            cross_validation=hyperparameters_optimizer_pb2.Evaluation.CrossValidation(
                fold_generator=fold_generator_pb2.FoldGenerator.CrossValidation(
                    num_folds=2
                ),
            ),
        ),
        search_space=hyperparameter_pb2.HyperParameterSpace(
            fields=[
                Field(
                    name="a",
                    discrete_candidates=DiscreteCandidates(
                        possible_values=[
                            Value(integer=1),
                            Value(integer=2),
                            Value(integer=3),
                        ],
                    ),
                ),
                Field(
                    name="b",
                    discrete_candidates=DiscreteCandidates(
                        possible_values=[
                            Value(real=1),
                            Value(real=2),
                            Value(real=3),
                        ]
                    ),
                ),
                Field(
                    name="c",
                    discrete_candidates=DiscreteCandidates(
                        possible_values=[
                            Value(categorical="x"),
                            Value(categorical="y"),
                            Value(categorical="v"),
                            Value(categorical="w"),
                        ]
                    ),
                    children=[
                        Field(
                            name="d",
                            discrete_candidates=DiscreteCandidates(
                                possible_values=[
                                    Value(integer=1),
                                    Value(integer=2),
                                    Value(integer=3),
                                ]
                            ),
                            parent_discrete_values=DiscreteCandidates(
                                possible_values=[
                                    Value(categorical="v"),
                                    Value(categorical="w"),
                                ],
                            ),
                        ),
                    ],
                ),
            ],
        ),
    )

    expected_random_optimizer = random_pb2.RandomOptimizerConfig(num_trials=20)

    # Note: Extension construction is not supported.
    expected_optimizer.optimizer.Extensions[random_pb2.random].CopyFrom(
        expected_random_optimizer
    )
    expected_proto.Extensions[
        hyperparameters_optimizer_pb2.hyperparameters_optimizer_config
    ].CopyFrom(expected_optimizer)

    test_utils.assertProto2Equal(self, tuner.train_config, expected_proto)

  def test_tuner_monotonic_gbt(self):
    dataset = test_utils.load_datasets("adult")

    tuner = tuner_lib.RandomSearchTuner(num_trials=5)
    tuner.choice("num_candidate_attributes_ratio", [1.0, 0.8, 0.6])
    tuner.choice("shrinkage", [0.05, 0.1, 0.2])

    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        tuner=tuner,
        num_trees=10,
        use_hessian_gain=True,
        features=[
            dataspec.Column("age", monotonic=+1),
            dataspec.Column("hours_per_week", monotonic=-1),
            dataspec.Column("education_num", monotonic=+1),
        ],
        include_all_columns=True,
    )

    model = learner.train(dataset.train_pd)
    self.assertIsNotNone(model)
    self.assertIsNotNone(model.hyperparameter_optimizer_logs())
    self.assertLen(model.hyperparameter_optimizer_logs().trials, 5)

  def test_tuner_monotonic_rf_fail(self):
    tuner = tuner_lib.RandomSearchTuner(num_trials=5)

    with self.assertRaisesRegex(
        test_utils.AbslInvalidArgumentError,
        "does not support monotonic constraints",
    ):
      _ = specialized_learners.RandomForestLearner(
          label="income",
          tuner=tuner,
          num_trees=10,
          features=[
              dataspec.Column("age", monotonic=+1),
          ],
          include_all_columns=True,
      )

  def test_optimize_metric(self):
    tuner = tuner_lib.RandomSearchTuner(optimize_metric="accuracy")
    tuner._set_task(generic_model.Task.CLASSIFICATION)
    self.assertTrue(tuner.train_config.Extensions[
        hyperparameters_optimizer_pb2.hyperparameters_optimizer_config
    ].evaluation.metric.classification.HasField("accuracy"))

  def test_optimize_metric_invalid_task(self):
    tuner = tuner_lib.RandomSearchTuner(optimize_metric="ACCURACY")
    with self.assertRaisesRegex(
        ValueError, "Metric ACCURACY is not compatible with task REGRESSION"
    ):
      tuner._set_task(generic_model.Task.REGRESSION)

  def test_optimize_metric_invalid(self):
    with self.assertRaisesRegex(
        ValueError, "Unknown metric 'invalid'. Supported metrics are:"
    ):
      tuner_lib.RandomSearchTuner(optimize_metric="invalid")

  def test_optimize_metric_binary_only(self):
    tuner = tuner_lib.RandomSearchTuner(optimize_metric="auc")
    tuner._set_task(generic_model.Task.CLASSIFICATION)
    data_spec = data_spec_pb2.DataSpecification()
    col = data_spec.columns.add()
    col.name = "label"
    col.categorical.number_of_unique_values = 4
    with self.assertRaisesRegex(
        ValueError, "only compatible with binary classification"
    ):
      tuner._validate_data_spec("label", data_spec, raise_error=True)

    col.categorical.number_of_unique_values = 3
    tuner._validate_data_spec(
        "label", data_spec, raise_error=True
    )  # Should not raise

if __name__ == "__main__":
  absltest.main()
