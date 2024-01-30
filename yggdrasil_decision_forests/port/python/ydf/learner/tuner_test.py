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

from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.learner.hyperparameters_optimizer import hyperparameters_optimizer_pb2
from yggdrasil_decision_forests.learner.hyperparameters_optimizer.optimizers import random_pb2
from yggdrasil_decision_forests.model import hyperparameter_pb2
from ydf.learner import tuner as tuner_lib
from ydf.utils import test_utils

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


if __name__ == "__main__":
  absltest.main()
