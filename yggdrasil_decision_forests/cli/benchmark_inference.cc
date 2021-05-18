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
#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
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

// Result from a single run.
struct Result {
  std::string name;
  absl::Duration avg_inference_duration;
};

// How to run the benchmark.
struct RunOptions {
  int num_runs;
  int batch_size;
  int warmup_runs;
};

std::string ResultsToString(const RunOptions& options,
                            std::vector<Result> results) {
  std::string report;

  // Sort the result from the fastest to the slowest.
  std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
    return a.avg_inference_duration < b.avg_inference_duration;
  });

  absl::StrAppendFormat(&report, "batch_size : %d  num_runs : %d\n",
                        options.batch_size, options.num_runs);
  absl::StrAppendFormat(&report, "time/example(us)  time/batch(us)  method\n");
  absl::StrAppendFormat(&report, "----------------------------------------\n");
  for (const auto& result : results) {
    absl::StrAppendFormat(
        &report, "%16.5g  %14.5g  %s\n",
        absl::ToDoubleMicroseconds(result.avg_inference_duration),
        absl::ToDoubleMicroseconds(result.avg_inference_duration *
                                   options.batch_size),
        result.name);
  }
  absl::StrAppendFormat(&report, "----------------------------------------\n");
  return report;
}

absl::Status BenchmarkGenericSlowEngine(const RunOptions& options,
                                        const model::AbstractModel& model,
                                        const dataset::VerticalDataset& dataset,
                                        std::vector<Result>* results) {
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
          LOG(INFO) << "Non supported task";
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
    const RunOptions& options, const model::FastEngineFactory& engine_factory,
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    std::vector<Result>* results) {
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

  const RunOptions options{
      /*.num_runs =*/absl::GetFlag(FLAGS_num_runs),
      /*.batch_size =*/absl::GetFlag(FLAGS_batch_size),
      /*.warmup_runs =*/absl::GetFlag(FLAGS_warmup_runs),
  };

  LOG(INFO) << "Loading model";
  std::unique_ptr<model::AbstractModel> model;
  RETURN_IF_ERROR(model::LoadModel(model_path, &model));
  LOG(INFO) << "The model is of type: " << model->name();

  LOG(INFO) << "Loading dataset";
  dataset::VerticalDataset dataset;
  RETURN_IF_ERROR(
      LoadVerticalDataset(dataset_path, model->data_spec(), &dataset));

  std::vector<Result> results;

  // Run engines.
  const auto engine_factories = model->ListCompatibleFastEngines();
  LOG(INFO) << "Found " << engine_factories.size()
            << " compatible fast engines.";
  for (const auto& engine_factory : engine_factories) {
    LOG(INFO) << "Running " << engine_factory->name();
    RETURN_IF_ERROR(BenchmarkFastEngineWithVirtualInterface(
        options, *engine_factory, *model.get(), dataset, &results));
  }

  if (absl::GetFlag(FLAGS_generic)) {
    LOG(INFO) << "Running the slow generic engine";
    RETURN_IF_ERROR(
        BenchmarkGenericSlowEngine(options, *model, dataset, &results));
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
    LOG(INFO) << "The benchmark failed with the following error: " << status;
    return 1;
  }
  return 0;
}
