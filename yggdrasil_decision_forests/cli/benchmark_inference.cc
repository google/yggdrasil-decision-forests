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

// Benchmarks the inference time of a model with the available inference
// engines.
//
// A model can be run with the "generic engine", a simple but slow
// implementation of the inference of the model that is always available and
// used as the golden implementation.
//
// Additional, "fast generic engines" are available with
// ListCompatibleFastEngines() or the "show_model" CLI. The availability of
// these engines depend on the structure of the model, the compile options and
// possibly the hardware. Note that engines should be linked to the binary.
//
// "fast generic engines" are accessible through a generic virtual interface, or
// by calling them directly. The second option being faster but require to know
// what interface is compatible at compilation time.
//
// The benchmark measures one copy of the dataset (from the best possible
// existing format; a simple memory copy in the best case) and one run of the
// engine.
//
// Usage example:
//
//   bazel run -c opt --copt=-mavx2 :benchmark_inference -- \
//     --alsologtostderr \
//     --model=/path/to/my/model \
//     --dataset=csv:/path/to/my/dataset.csv
//
// Result:
//
//   batch_size : 100  num_runs : 20
//   time/example(µs)  time/batch(µs)  method
//   ----------------------------------------
//   0.79025         79.025  GradientBoostedTreesQuickScorerExtended
//   9.179           917.9   GradientBoostedTreesGeneric
//   21.547          2154.8  Generic slow engine
//   ----------------------------------------
//
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/benchmark/inference.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, model, "", "Path to model.");
ABSL_FLAG(std::string, dataset, "",
          "Typed path to dataset i.e. [type]:[path] format.");
ABSL_FLAG(
    int, num_runs, 20,
    "Number of times the dataset is run. Higher values increase the "
    "precision of the timings, but increase the duration of the benchmark.");
ABSL_FLAG(int, batch_size, 100,
          "Number of examples per batch. Note that some engine are not impacted"
          "by the batch size.");
ABSL_FLAG(int, warmup_runs, 1,
          "Number of runs through the dataset before the benchmark.");
ABSL_FLAG(bool, generic, true,
          "Evaluates the slow engine i.e. model->predict(). The "
          "generic engine is slow and mostly a reference. Disable it if the "
          "benchmark runs for too long.");

constexpr char kUsageMessage[] =
    "Benchmarks the inference time of a model with the available inference "
    "engines.";

namespace yggdrasil_decision_forests {

std::string ResultsToString(
    const utils::BenchmarkInferenceRunOptions& options,
    std::vector<utils::BenchmarkInferenceResult> results) {
  std::string report;

  // Sort the result from the fastest to the slowest.
  std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
    return a.duration_per_example < b.duration_per_example;
  });

  absl::StrAppendFormat(&report, "batch_size : %d  num_runs : %d\n",
                        options.batch_size, options.runs->num_runs);
  absl::StrAppendFormat(&report, "time/example(us)  time/batch(us)  method\n");
  absl::StrAppendFormat(&report, "----------------------------------------\n");
  for (const auto& result : results) {
    absl::StrAppendFormat(
        &report, "%16.5g  %14.5g  %s\n",
        absl::ToDoubleMicroseconds(result.duration_per_example),
        absl::ToDoubleMicroseconds(result.duration_per_example *
                                   options.batch_size),
        result.name);
  }
  absl::StrAppendFormat(&report, "----------------------------------------\n");
  return report;
}

absl::Status Benchmark() {
  // Parse flags.
  const auto model_path = absl::GetFlag(FLAGS_model);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("The --model is not specified.");
  }

  const auto dataset_path = absl::GetFlag(FLAGS_dataset);
  if (dataset_path.empty()) {
    return absl::InvalidArgumentError("The --dataset is not specified.");
  }
  const utils::BenchmarkInterfaceNumRunsOptions num_runs_options = {
      /*.num_runs =*/absl::GetFlag(FLAGS_num_runs),
      /*.warmup_runs =*/absl::GetFlag(FLAGS_warmup_runs),
  };
  const utils::BenchmarkInferenceRunOptions options{
      /*.batch_size =*/absl::GetFlag(FLAGS_batch_size),
      /*.runs =*/num_runs_options,
      /*.time =*/absl::nullopt};

  YDF_LOG(INFO) << "Loading model";
  std::unique_ptr<model::AbstractModel> model;
  RETURN_IF_ERROR(model::LoadModel(model_path, &model));
  YDF_LOG(INFO) << "The model is of type: " << model->name();

  YDF_LOG(INFO) << "Loading dataset";
  dataset::VerticalDataset dataset;
  RETURN_IF_ERROR(
      LoadVerticalDataset(dataset_path, model->data_spec(), &dataset,
                          /*ensure_non_missing=*/model->input_features()));

  std::vector<utils::BenchmarkInferenceResult> results;

  // Run engines.
  const auto engine_factories = model->ListCompatibleFastEngines();
  YDF_LOG(INFO) << "Found " << engine_factories.size()
                << " compatible fast engines.";
  for (const auto& engine_factory : engine_factories) {
    YDF_LOG(INFO) << "Running " << engine_factory->name();
    ASSIGN_OR_RETURN(auto engine, engine_factory->CreateEngine(model.get()));
    RETURN_IF_ERROR(utils::BenchmarkFastEngine(options, *engine.get(),
                                               *model.get(), dataset, &results,
                                               engine_factory->name()));
  }

  if (absl::GetFlag(FLAGS_generic)) {
    YDF_LOG(INFO) << "Running the slow generic engine";
    RETURN_IF_ERROR(
        utils::BenchmarkGenericSlowEngine(options, *model, dataset, &results));
  }

  // Show results.
  std::cout << ResultsToString(options, results);
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  const auto status = yggdrasil_decision_forests::Benchmark();
  if (!status.ok()) {
    YDF_LOG(INFO) << "The benchmark failed with the following error: "
                  << status;
    return 1;
  }
  return 0;
}
