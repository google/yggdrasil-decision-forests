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

"""Tests for the custom losses."""

from absl.testing import absltest
from absl.testing import parameterized
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from ydf.learner import hyperparameters as hp_lib


class HyperparametersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="no_mutual_exclusive",
          hp_dict={"max_depth": 3, "num_trees": 20},
      ),
      dict(
          testcase_name="single_mutual_exclusive",
          hp_dict={"max_depth": 3, "num_candidate_attributes": 20},
      ),
      dict(
          testcase_name="none_mutual_exclusive",
          hp_dict={
              "num_candidate_attributes_ratio": None,
              "num_candidate_attributes": 20,
          },
      ),
  )
  def test_validate_hyperparameters(self, hp_dict):
    train_config = abstract_learner_pb2.TrainingConfig(
        learner="RANDOM_FOREST", label="label"
    )
    hp_lib.validate_hyperparameters(
        hp_dict=hp_dict,
        train_config=train_config,
        deployment_config=abstract_learner_pb2.DeploymentConfig(),
    )

  def test_validate_hyperparameters_fails(self):
    train_config = abstract_learner_pb2.TrainingConfig(
        learner="RANDOM_FOREST", label="label"
    )
    hp_dict = {
        "num_candidate_attributes_ratio": 1.0,
        "num_candidate_attributes": 20,
    }
    with self.assertRaisesRegex(
        ValueError,
        ".*Only one of the following hyperparameters can be set:"
        " (num_candidate_attributes,"
        " num_candidate_attributes_ratio|num_candidate_attributes_ratio,"
        " num_candidate_attributes).*",
    ):
      hp_lib.validate_hyperparameters(
          hp_dict=hp_dict,
          train_config=train_config,
          deployment_config=abstract_learner_pb2.DeploymentConfig(),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="prune_first_candidate_solo",
          hp_dict={
              "subsample_ratio": 1.0,
              "subsample_count": 20,
          },
          explicit_parameters=set(["subsample_ratio"]),
          expected={"subsample_ratio": 1.0},
      ),
      dict(
          testcase_name="prune_second_candidate_solo",
          hp_dict={
              "subsample_ratio": 1.0,
              "subsample_count": 20,
          },
          explicit_parameters=set(["subsample_count"]),
          expected={"subsample_count": 20},
      ),
      dict(
          testcase_name="prune_candidate_with_explicit",
          hp_dict={
              "subsample_ratio": 1.0,
              "subsample_count": 20,
              "max_depth": 5,
          },
          explicit_parameters=set(["subsample_count", "max_depth"]),
          expected={
              "subsample_count": 20,
              "max_depth": 5,
          },
      ),
      dict(
          testcase_name="prune_candidate_with_nonexplicit",
          hp_dict={
              "subsample_ratio": 1.0,
              "subsample_count": 20,
              "max_depth": 5,
          },
          explicit_parameters=set(["subsample_count"]),
          expected={
              "subsample_count": 20,
              "max_depth": 5,
          },
      ),
      dict(
          testcase_name="two_mutual_exlusive",
          hp_dict={
              "subsample_ratio": 1.0,
              "subsample_count": 20,
              "subsample_ratio": 0.5,
              "subsample_count": 5,
          },
          explicit_parameters=set(["subsample_count", "subsample_count"]),
          expected={
              "subsample_count": 20,
              "subsample_count": 5,
          },
      ),
  )
  def test_fix_hyperparameters(self, hp_dict, explicit_parameters, expected):
    train_config = abstract_learner_pb2.TrainingConfig(
        learner="ISOLATION_FOREST", label="label"
    )
    got = hp_lib.fix_hyperparameters(
        hp_dict=hp_dict,
        explicit_parameters=explicit_parameters,
        train_config=train_config,
        deployment_config=abstract_learner_pb2.DeploymentConfig(),
    )
    self.assertEqual(expected, got)

  def test_fix_hyperparameters_fails(self):
    train_config = abstract_learner_pb2.TrainingConfig(
        learner="ISOLATION_FOREST", label="label"
    )
    hp_dict = {
        "subsample_ratio": 1.0,
        "subsample_count": 20,
    }
    explicit_parameters = set(["subsample_count", "subsample_ratio"])
    with self.assertRaisesRegex(
        ValueError,
        ".*Only one of the following hyperparameters can be set:"
        " (subsample_count,"
        " subsample_ratio|subsample_ratio,"
        " subsample_count).*",
    ):
      _ = hp_lib.fix_hyperparameters(
          hp_dict=hp_dict,
          explicit_parameters=explicit_parameters,
          train_config=train_config,
          deployment_config=abstract_learner_pb2.DeploymentConfig(),
      )


if __name__ == "__main__":
  absltest.main()
