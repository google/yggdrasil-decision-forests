/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/hyperparameters_optimizer.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

class AutotunedGradientBoostedTreesOnAdult : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_ = PARSE_TEST_PROTO(R"pb(
      task: CLASSIFICATION
      learner: "HYPERPARAMETER_OPTIMIZER"
      label: "income"

      [yggdrasil_decision_forests.model.hyperparameters_optimizer_v2.proto
           .hyperparameters_optimizer_config] {

        retrain_final_model: true

        optimizer {
          optimizer_key: "RANDOM"
          [yggdrasil_decision_forests.model.hyperparameters_optimizer_v2.proto
               .random] { num_trials: 25 }
        }

        base_learner {
          learner: "GRADIENT_BOOSTED_TREES"
          [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
               .gradient_boosted_trees_config] { num_trees: 50 }
        }

        base_learner_deployment {
          # The multi-threading is done at the optimizer level.
          num_threads: 1
        }

        predefined_search_space {}
      }
    )pb");

    train_config_.set_learner(
        hyperparameters_optimizer_v2::HyperParameterOptimizerLearner::
            kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("income");
    train_config_.add_features(".*");
    dataset_filename_ = "adult.csv";

    deployment_config_.set_cache_path(
        file::JoinPath(test::TmpDirectory(), "working_directory"));
  }
};

TEST_F(AutotunedGradientBoostedTreesOnAdult,
       RandomTuner_MemoryDataset_LocalTraining) {
  TrainAndEvaluateModel();
  EXPECT_GE(metric::Accuracy(evaluation_), 0.87);
  EXPECT_LT(metric::LogLoss(evaluation_), 0.30);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 25);
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
