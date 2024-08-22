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
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

#include "examples/hyperparameter_sweep/optimizer.pb.h"

namespace ydf = ::yggdrasil_decision_forests;
namespace distribute = ::ydf::distribute;

namespace example {

struct BenchmarkResult {
  // Returns the average inference time per prediction.
  double time_per_predictions_s = 0;
};

// How to run the inference benchmark.
struct RunOptions {
  int num_runs = 5;
  int batch_size = 200;
  int warmup_runs = 1;
};

absl::StatusOr<BenchmarkResult> BenchmarkCPUInferenceSpeed(
    const RunOptions& options, const ydf::model::AbstractModel& model,
    const ydf::dataset::VerticalDataset& dataset) {
  // Compile the model.

  ASSIGN_OR_RETURN(const auto engine, model.BuildFastEngine());
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
  return BenchmarkResult{
      .time_per_predictions_s = absl::ToDoubleSeconds(
          (end_time - start_time) / (options.num_runs * dataset.nrow()))};
}

class Worker : public distribute::AbstractWorker {
 public:
  absl::Status Setup(distribute::Blob serialized_welcome) override {
    ASSIGN_OR_RETURN(init_message,
                     ydf::utils::ParseBinaryProto<proto::Initialization>(
                         serialized_welcome));
    return absl::OkStatus();
  }

  absl::StatusOr<distribute::Blob> RunRequest(
      distribute::Blob serialized_request) override {
    ASSIGN_OR_RETURN(auto request, ydf::utils::ParseBinaryProto<proto::Request>(
                                       serialized_request));

    LOG(INFO) << "Create dataspec";
    ydf::dataset::proto::DataSpecification dataspec;
    ydf::dataset::CreateDataSpec(init_message.train_path(), false,
                                 request.guide(), &dataspec);

    LOG(INFO) << "Train model";
    std::unique_ptr<ydf::model::AbstractLearner> learner;
    RETURN_IF_ERROR(GetLearner(request.train_config(), &learner));

    proto::Result result;
    for (int repetition_idx = 0; repetition_idx < request.num_repetitions();
         repetition_idx++) {
      auto& result_item = *result.add_items();
      result_item.set_repetition_idx(repetition_idx);
      result_item.set_run_idx(request.run_idx());

      learner->mutable_training_config()->set_random_seed(1 + repetition_idx);

      ASSIGN_OR_RETURN(auto model, learner->TrainWithStatus(
                                       init_message.train_path(), dataspec));

      auto* gbt_model = dynamic_cast<
          const ydf::model::gradient_boosted_trees::GradientBoostedTreesModel*>(
          model.get());
      STATUS_CHECK(gbt_model);

      LOG(INFO) << "Load test dataset";
      ydf::dataset::VerticalDataset test_dataset;
      RETURN_IF_ERROR(ydf::dataset::LoadVerticalDataset(
          init_message.test_path(), model->data_spec(), &test_dataset));

      LOG(INFO) << "Evaluate model";
      ydf::utils::RandomEngine rnd;
      ydf::metric::proto::EvaluationOptions eval_option;
      eval_option.set_bootstrapping_samples(-1);
      ASSIGN_OR_RETURN(
          const auto evaluation,
          model->EvaluateWithStatus(test_dataset, eval_option, &rnd));

      *result_item.mutable_param_json() = absl::Substitute(
          R"($0
"effective_num_trees": $1,
"effective_num_nodes": $2,)",
          request.param_json(), gbt_model->num_trees(), gbt_model->NumNodes());
      result_item.set_accuracy(ydf::metric::Accuracy(evaluation));
      for (const auto& roc : evaluation.classification().rocs()) {
        result_item.add_aucs(roc.auc());
      }

      // Measure model speed
      ASSIGN_OR_RETURN(
          const auto speed_cpu_benchmark,
          BenchmarkCPUInferenceSpeed(RunOptions{}, *model.get(), test_dataset));
      result_item.set_time_per_predictions_s(
          speed_cpu_benchmark.time_per_predictions_s);
    }

    return result.SerializeAsString();
  }

 private:
  proto::Initialization init_message;
};

}  // namespace example

namespace yggdrasil_decision_forests {
namespace distribute {
using Worker = example::Worker;
REGISTER_Distribution_Worker(Worker, "HYPER_PARAMETER_SWEEPER");
}  // namespace distribute
}  // namespace yggdrasil_decision_forests
