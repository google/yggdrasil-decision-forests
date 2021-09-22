/*
 * Copyright 2021 Google LLC.
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

#include "yggdrasil_decision_forests/learner/generic_worker/generic_worker.h"

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/learner/generic_worker/generic_worker.pb.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/multi_thread/multi_thread.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace generic_worker {
namespace {

// Create a single thread manager with 5 workers.
std::unique_ptr<distribute::AbstractManager> CreateSingleThreadManager() {
  distribute::proto::Config config;
  config.set_implementation_key("MULTI_THREAD");
  config.MutableExtension(distribute::proto::multi_thread)->set_num_workers(5);
  config.set_verbosity(2);
  proto::Welcome welcome;
  welcome.set_temporary_directory(test::TmpDirectory());
  return distribute::CreateManager(config,
                                   /*worker_name=*/GenericWorker::kWorkerKey,
                                   /*welcome_blob=*/welcome.SerializeAsString())
      .value();
}

TEST(TrainAndEvaluateModel, Base) {
  auto manager = CreateSingleThreadManager();

  proto::Request train_request;
  {
    auto& train_model = *train_request.mutable_train_model();
    *train_model.mutable_train_config() = PARSE_TEST_PROTO(
        R"pb(
          learner: "RANDOM_FOREST"
          label: "income"
          [yggdrasil_decision_forests.model.random_forest.proto
               .random_forest_config] {
            winner_take_all_inference: false
            num_trees: 50
          }
        )pb");
    train_model.set_dataset_path(absl::StrCat(
        "csv:", file::JoinPath(
                    test::DataRootDirectory(),
                    "yggdrasil_decision_forests/test_data/dataset/"
                    "adult_train.csv")));

    dataset::CreateDataSpec(train_model.dataset_path(), false, {},
                            train_model.mutable_dataspec());
    train_model.set_model_base_path(
        file::JoinPath(test::TmpDirectory(), "my_model"));
  }

  auto train_result =
      manager->BlockingProtoRequest<proto::Result>(train_request).value();

  proto::Request evaluate_request;
  {
    auto& evaluate_model = *evaluate_request.mutable_evaluate_model();
    evaluate_model.set_model_path(train_result.train_model().model_path());
    evaluate_model.set_dataset_path(absl::StrCat(
        "csv:", file::JoinPath(
                    test::DataRootDirectory(),
                    "yggdrasil_decision_forests/test_data/dataset/"
                    "adult_test.csv")));
  }

  auto evaluate_result =
      manager->BlockingProtoRequest<proto::Result>(evaluate_request).value();

  LOG(INFO) << "Evaluation:"
            << metric::TextReport(
                   evaluate_result.evaluate_model().evaluation());

  EXPECT_GE(metric::Accuracy(evaluate_result.evaluate_model().evaluation()),
            0.86);

  EXPECT_OK(manager->Done());
}

}  // namespace
}  // namespace generic_worker
}  // namespace model
}  // namespace yggdrasil_decision_forests