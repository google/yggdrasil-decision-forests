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

#include "yggdrasil_decision_forests/utils/benchmark/inference.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::utils {
namespace {

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

TEST(BenchmarkInference, FastEngine) {
  std::unique_ptr<model::AbstractModel> model;
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_rf"), &model));

  dataset::VerticalDataset dataset;
  EXPECT_OK(dataset::LoadVerticalDataset(
      absl::StrCat("csv:",
                   file::JoinPath(TestDataDir(), "dataset", "adult_test.csv")),
      model->data_spec(), &dataset));
  const BenchmarkInterfaceNumRunsOptions num_runs_options = {
      /*.num_runs =*/5,
      /*.warmup_runs =*/1,
  };
  const BenchmarkInferenceRunOptions options{/*.batch_size =*/2,
                                             /*.runs =*/num_runs_options,
                                             /*.time =*/std::nullopt};
  std::vector<BenchmarkInferenceResult> results;

  ASSERT_OK_AND_ASSIGN(auto engine, model->BuildFastEngine());
  ASSERT_OK(
      BenchmarkFastEngine(options, *engine.get(), *model, dataset, &results));
  ASSERT_THAT(results, testing::SizeIs(1));
  EXPECT_GT(absl::ToDoubleSeconds(results[0].duration_per_example), 0);
}

TEST(BenchmarkInference, GenericEngine) {
  const BenchmarkInterfaceNumRunsOptions num_runs_options = {
      /*.num_runs =*/5,
      /*.warmup_runs =*/1,
  };
  const BenchmarkInferenceRunOptions options{/*.batch_size =*/2,
                                             /*.runs =*/num_runs_options,
                                             /*.time =*/std::nullopt};
  std::vector<BenchmarkInferenceResult> results;

  std::unique_ptr<model::AbstractModel> model;
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_rf"), &model));

  dataset::VerticalDataset dataset;
  EXPECT_OK(dataset::LoadVerticalDataset(
      absl::StrCat("csv:",
                   file::JoinPath(TestDataDir(), "dataset", "adult_test.csv")),
      model->data_spec(), &dataset));

  ASSERT_OK(BenchmarkGenericSlowEngine(options, *model, dataset, &results));
  ASSERT_THAT(results, testing::SizeIs(1));
  EXPECT_GT(absl::ToDoubleSeconds(results[0].duration_per_example), 0);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::utils
