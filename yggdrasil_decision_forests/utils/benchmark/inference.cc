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
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/fast_engine_factory.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::utils {

namespace {
void RunOnceGeneric(const model::AbstractModel& model,
                    const dataset::VerticalDataset& dataset,
                    std::vector<float>& predictions) {
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
}

void RunOnceEngine(const serving::FastEngine& engine, const int num_batches,
                   const int batch_size, const int64_t total_num_examples,
                   serving::AbstractExampleSet* examples,
                   serving::AbstractExampleSet* batch_of_examples,
                   const serving::FeaturesDefinition& engine_features,
                   std::vector<float>& predictions) {
  for (int64_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    const int64_t begin_example_idx = batch_idx * batch_size;
    const int64_t end_example_idx =
        std::min(begin_example_idx + batch_size, total_num_examples);
    // Copy the examples.
    CHECK_OK(examples->Copy(begin_example_idx, end_example_idx, engine_features,
                            batch_of_examples));
    // Runs the engine.
    engine.Predict(*batch_of_examples, end_example_idx - begin_example_idx,
                   &predictions);
  }
}

}  // namespace

absl::Status BenchmarkGenericSlowEngine(
    const BenchmarkInferenceRunOptions& options,
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    std::vector<BenchmarkInferenceResult>* results) {
  if (options.time.has_value() == options.runs.has_value()) {
    return absl::InvalidArgumentError(
        "Specify either the number of runs or the timing of the benchmark.");
  }
  std::vector<float> predictions(dataset.nrow());

  int num_benchmark_runs;
  // Warming up.
  if (options.runs.has_value()) {
    num_benchmark_runs = options.runs->warmup_runs;
    for (int run_idx = 0; run_idx < options.runs->warmup_runs; run_idx++) {
      RunOnceGeneric(model, dataset, predictions);
    }
  } else {
    STATUS_CHECK_GT(options.time->warmup_duration, 0);
    STATUS_CHECK_GT(options.time->benchmark_duration, 0);
    const auto warmup_start = absl::Now();
    const auto warmup_cutoff =
        warmup_start + absl::Seconds(options.time->warmup_duration);
    int num_warmup_runs = 0;
    while (absl::Now() < warmup_cutoff) {
      RunOnceGeneric(model, dataset, predictions);
      ++num_warmup_runs;
    }
    const auto actual_warmup_duration = absl::Now() - warmup_start;
    STATUS_CHECK_GT(num_warmup_runs, 0);
    double estimated_seconds_per_run =
        absl::ToDoubleSeconds(actual_warmup_duration) / num_warmup_runs;
    num_benchmark_runs =
        std::ceil(options.time->benchmark_duration / estimated_seconds_per_run);
  }

  // Run benchmark.
  const auto start_time = absl::Now();
  for (int run_idx = 0; run_idx < num_benchmark_runs; run_idx++) {
    RunOnceGeneric(model, dataset, predictions);
  }
  const auto end_time = absl::Now();

  // Save results.
  results->push_back(
      {/*.name =*/"Generic slow engine",
       /*.duration_per_example =*/
       (end_time - start_time) / (num_benchmark_runs * dataset.nrow()),
       /*benchmark_duration=*/end_time - start_time,
       /*num_runs=*/num_benchmark_runs,
       /*batch_size=*/options.batch_size});
  return absl::OkStatus();
}

absl::Status BenchmarkFastEngine(const BenchmarkInferenceRunOptions& options,
                                 const serving::FastEngine& engine,
                                 const model::AbstractModel& model,
                                 const dataset::VerticalDataset& dataset,
                                 std::vector<BenchmarkInferenceResult>* results,
                                 absl::string_view engine_name) {
  if (options.time.has_value() == options.runs.has_value()) {
    return absl::InvalidArgumentError(
        "Specify either the number of runs or the timing of the benchmark.");
  }
  const auto& engine_features = engine.features();

  // Convert dataset into the format expected by the engine.
  const int64_t total_num_examples = dataset.nrow();
  auto examples = engine.AllocateExamples(total_num_examples);
  STATUS_CHECK(CopyVerticalDatasetToAbstractExampleSet(
                   dataset,
                   /*begin_example_idx=*/0,
                   /*end_example_idx=*/total_num_examples, engine_features,
                   examples.get())
                   .ok());

  // Allocate a batch of examples.
  auto batch_of_examples = engine.AllocateExamples(options.batch_size);
  const int64_t num_batches =
      (total_num_examples + options.batch_size - 1) / options.batch_size;

  std::vector<float> predictions(dataset.nrow());

  int num_benchmark_runs;

  // Warming up.
  // TODO: Simplify this code.
  if (options.runs.has_value()) {
    const auto warmup_runs = options.runs->warmup_runs;
    for (int run_idx = 0; run_idx < warmup_runs; run_idx++) {
      RunOnceEngine(engine, num_batches, options.batch_size, total_num_examples,
                    examples.get(), batch_of_examples.get(), engine_features,
                    predictions);
    }
    num_benchmark_runs = options.runs->num_runs;
  } else {
    STATUS_CHECK_GT(options.time->warmup_duration, 0);
    STATUS_CHECK_GT(options.time->benchmark_duration, 0);
    const auto warmup_start = absl::Now();
    const auto warmup_cutoff =
        warmup_start + absl::Seconds(options.time->warmup_duration);
    int num_warmup_runs = 0;

    while (absl::Now() < warmup_cutoff) {
      RunOnceEngine(engine, num_batches, options.batch_size, total_num_examples,
                    examples.get(), batch_of_examples.get(), engine_features,
                    predictions);
      ++num_warmup_runs;
    }

    STATUS_CHECK_GT(num_warmup_runs, 0);
    auto estimated_seconds_per_run =
        options.time->warmup_duration / num_warmup_runs;
    num_benchmark_runs =
        std::ceil(options.time->benchmark_duration / estimated_seconds_per_run);
  }

  // Run benchmark.
  const auto start_time = absl::Now();
  for (int run_idx = 0; run_idx < num_benchmark_runs; run_idx++) {
    RunOnceEngine(engine, num_batches, options.batch_size, total_num_examples,
                  examples.get(), batch_of_examples.get(), engine_features,
                  predictions);
  }
  const auto end_time = absl::Now();

  // Save results.
  results->push_back(
      {/*.name =*/absl::StrCat(engine_name, " [virtual interface]"),
       /*.duration_per_example =*/
       (end_time - start_time) / (num_benchmark_runs * dataset.nrow()),
       /*benchmark_duration=*/end_time - start_time,
       /*num_runs=*/num_benchmark_runs,
       /*batch_size=*/options.batch_size});
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::utils
