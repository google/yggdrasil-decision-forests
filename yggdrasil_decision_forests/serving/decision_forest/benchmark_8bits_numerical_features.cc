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

// This tool measures the inference speed of te 8bits numerical feature engine.
//
// The signature of this tool is similar to the generic
// "cli/benchmark_inference" benchmark inference tool.
//
#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/decision_forest/8bits_numerical_features.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

ABSL_FLAG(std::string, model, "",
          "Path to model. Should be a model compatible with the 8bits engine.");
ABSL_FLAG(std::string, dataset, "",
          "Typed path to dataset i.e. [type]:[path] format.");
ABSL_FLAG(
    int, num_runs, 100,
    "Number of times the dataset is run. Higher values increase the "
    "precision of the timings, but increase the duration of the benchmark.");
ABSL_FLAG(int, warmup_runs, 20,
          "Number of runs through the dataset before the benchmark.");

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {
namespace num_8bits {

absl::Status Benchmark() {
  YDF_LOG(INFO) << "Load model";
  std::unique_ptr<model::AbstractModel> model;
  RETURN_IF_ERROR(model::LoadModel(absl::GetFlag(FLAGS_model), &model));

  YDF_LOG(INFO) << "Load dataset";
  dataset::VerticalDataset dataset;
  RETURN_IF_ERROR(LoadVerticalDataset(absl::GetFlag(FLAGS_dataset),
                                      model->data_spec(), &dataset));

  YDF_LOG(INFO) << "Compile model";
  // Compile model
  auto* gbt_model =
      dynamic_cast<model::gradient_boosted_trees::GradientBoostedTreesModel*>(
          model.get());
  STATUS_CHECK(gbt_model);

  // The regressive engine is compatible with all models that output a single
  // value (e.g. ranking, regression, binary classification). Using the
  // regressive engine on binary classification models is more efficient than
  // using the GradientBoostedTreesBinaryClassificationModel engine as the
  // regresive engine skips the final activation function (which can be a
  // significant part of the inference cost).
  GradientBoostedTreesBinaryRegressiveModel engine;

  // The binary classification engine.
  // GradientBoostedTreesBinaryClassificationModel engine;

  RETURN_IF_ERROR(GenericToSpecializedModel(*gbt_model, &engine));

  YDF_LOG(INFO) << "Details:\n" << EngineDetails(engine);

  // Copy data to the format expected by the engine
  std::vector<uint8_t> examples(engine.num_features * dataset.nrow());
  for (int example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    for (int local_feature_idx = 0; local_feature_idx < engine.num_features;
         local_feature_idx++) {
      const auto feature_index_values =
          dataset
              .ColumnWithCast<
                  dataset::VerticalDataset::DiscretizedNumericalColumn>(
                  engine.features[local_feature_idx])
              ->values();

      // Note: For this model, the discretized feature indices and the
      // corresponding feature values are equal.
      examples[example_idx * engine.num_features + local_feature_idx] =
          feature_index_values[example_idx];
    }
  }
  std::vector<float> engine_predictions;

  YDF_LOG(INFO) << "Run benchmark";

  auto run_once = [&]() {
    Predict(engine, examples, dataset.nrow(), &engine_predictions)
        .IgnoreError();
  };

  // Warmup
  const auto warmup_runs = absl::GetFlag(FLAGS_warmup_runs);
  for (int run_idx = 0; run_idx < warmup_runs; run_idx++) {
    run_once();
  }

  // Run benchmark.
  const auto num_runs = absl::GetFlag(FLAGS_num_runs);
  const auto start_time = absl::Now();
  for (int run_idx = 0; run_idx < num_runs; run_idx++) {
    run_once();
  }
  const auto end_time = absl::Now();

  const auto time_per_example =
      (end_time - start_time) / (num_runs * dataset.nrow());
  YDF_LOG(INFO) << "Average inference time per example: " << time_per_example;

  return absl::OkStatus();
}

}  // namespace num_8bits
}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging("", &argc, &argv, true);
  const auto status = yggdrasil_decision_forests::serving::decision_forest::
      num_8bits::Benchmark();
  if (!status.ok()) {
    YDF_LOG(INFO) << "The benchmark failed with the following error: "
                  << status;
    return 1;
  }
  return 0;
}
