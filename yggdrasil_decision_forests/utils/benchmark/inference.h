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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_BENCHMARK_INFERENCE_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_BENCHMARK_INFERENCE_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"

namespace yggdrasil_decision_forests::utils {

// Result from a single run.
struct BenchmarkInferenceResult {
  std::string name;
  absl::Duration duration_per_example;
  absl::Duration benchmark_duration;
  int num_runs;
  int batch_size;
};

struct BenchmarkInterfaceNumRunsOptions {
  int num_runs;
  int warmup_runs;
};

struct BenchmarkInterfaceTimingOptions {
  double benchmark_duration;
  double warmup_duration;
};

// How to run the benchmark.
struct BenchmarkInferenceRunOptions {
  int batch_size;
  absl::optional<BenchmarkInterfaceNumRunsOptions> runs;
  absl::optional<BenchmarkInterfaceTimingOptions> time;
};

// Benchmark the inference time per example using the generic slow engine.
// The benchmark measures one copy of the dataset (from the best possible
// existing format; a simple memory copy in the best case) and one run of the
// engine.
absl::Status BenchmarkGenericSlowEngine(
    const BenchmarkInferenceRunOptions& options,
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    std::vector<BenchmarkInferenceResult>* results);

// Benchmark the inference time per example using the a fast engine.
// The benchmark measures one copy of the dataset (from the best possible
// existing format; a simple memory copy in the best case) and one run of the
// engine.
absl::Status BenchmarkFastEngine(const BenchmarkInferenceRunOptions& options,
                                 const serving::FastEngine& engine,
                                 const model::AbstractModel& model,
                                 const dataset::VerticalDataset& dataset,
                                 std::vector<BenchmarkInferenceResult>* results,
                                 absl::string_view engine_name = "");

}  // namespace yggdrasil_decision_forests::utils

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_BENCHMARK_INFERENCE_H_
