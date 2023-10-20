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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/fast_engine_factory.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests::utils {

absl::Status BenchmarkGenericSlowEngine(
    const BenchmarkInferenceRunOptions& options,
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    std::vector<BenchmarkInferenceResult>* results) {
  std::vector<float> predictions(dataset.nrow());

  auto run_once = [&]() {
    model::proto::Prediction prediction;
    for (dataset::VerticalDataset::row_t example_idx = 0;
         example_idx < dataset.nrow(); example_idx++) {
      model.Predict(dataset, example_idx, &prediction);
      switch (prediction.type_case()) {
        case model::proto::Prediction::kClassification:
          predictions[example_idx] =
              prediction.classification().distribution().counts(2) /
              prediction.classification().distribution().sum();
          break;
        case model::proto::Prediction::kRegression:
          predictions[example_idx] = prediction.regression().value();
          break;
        case model::proto::Prediction::kRanking:
          predictions[example_idx] = prediction.ranking().relevance();
          break;
        default:
          YDF_LOG(INFO) << "Non supported task";
      }
    }
  };

  // Warming up.
  for (int run_idx = 0; run_idx < options.warmup_runs; run_idx++) {
    run_once();
  }

  // Run benchmark.
  const auto start_time = absl::Now();
  for (int run_idx = 0; run_idx < options.num_runs; run_idx++) {
    run_once();
  }
  const auto end_time = absl::Now();

  // Save results.
  results->push_back(
      {/*.name =*/"Generic slow engine",
       /*.avg_inference_duration =*/
       (end_time - start_time) / (options.num_runs * dataset.nrow())});
  return absl::OkStatus();
}

absl::Status BenchmarkFastEngineWithVirtualInterface(
    const BenchmarkInferenceRunOptions& options,
    const model::FastEngineFactory& engine_factory,
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    std::vector<BenchmarkInferenceResult>* results) {
  // Compile the model.
  ASSIGN_OR_RETURN(const auto engine, engine_factory.CreateEngine(&model));
  const auto& engine_features = engine->features();

  // Convert dataset into the format expected by the engine.
  const int64_t total_num_examples = dataset.nrow();
  auto examples = engine->AllocateExamples(total_num_examples);
  CHECK_OK(CopyVerticalDatasetToAbstractExampleSet(
      dataset,
      /*begin_example_idx=*/0,
      /*end_example_idx=*/total_num_examples, engine_features, examples.get()));

  // Allocate a batch of examples.
  auto batch_of_examples = engine->AllocateExamples(options.batch_size);
  const int64_t num_batches =
      (total_num_examples + options.batch_size - 1) / options.batch_size;

  std::vector<float> predictions(dataset.nrow());

  auto run_once = [&]() {
    for (int64_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
      const int64_t begin_example_idx = batch_idx * options.batch_size;
      const int64_t end_example_idx =
          std::min(begin_example_idx + options.batch_size, total_num_examples);
      // Copy the examples.
      CHECK_OK(examples->Copy(begin_example_idx, end_example_idx,
                              engine_features, batch_of_examples.get()));
      // Runs the engine.
      engine->Predict(*batch_of_examples, end_example_idx - begin_example_idx,
                      &predictions);
    }
  };

  // Warming up.
  for (int run_idx = 0; run_idx < options.warmup_runs; run_idx++) {
    run_once();
  }

  // Run benchmark.
  const auto start_time = absl::Now();
  for (int run_idx = 0; run_idx < options.num_runs; run_idx++) {
    run_once();
  }
  const auto end_time = absl::Now();

  // Save results.
  results->push_back(
      {/*.name =*/absl::StrCat(engine_factory.name(), " [virtual interface]"),
       /*.avg_inference_duration =*/
       (end_time - start_time) / (options.num_runs * dataset.nrow())});
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::utils
