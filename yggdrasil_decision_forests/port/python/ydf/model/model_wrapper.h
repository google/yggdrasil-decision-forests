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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_LEARNER_MODEL_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_LEARNER_MODEL_H_

#include <pybind11/numpy.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/benchmark/inference.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/model_analysis.pb.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {

// This class is a pybind-compatible alternative to
// utils::BenchmarkInferenceResult which does not use absl::Duration objects.
struct BenchmarkInferenceCCResult {
  double duration_per_example;
  double benchmark_duration;
  int num_runs;
  int batch_size;

  BenchmarkInferenceCCResult(const utils::BenchmarkInferenceResult& result)
      : duration_per_example(
            absl::ToDoubleSeconds(result.duration_per_example)),
        benchmark_duration(absl::ToDoubleSeconds(result.benchmark_duration)),
        num_runs(result.num_runs),
        batch_size(result.batch_size) {}

  std::string ToString() const;
};

class GenericCCModel {
 public:
  GenericCCModel(std::unique_ptr<model::AbstractModel>&& model)
      : model_(std::move(model)) {}

  virtual ~GenericCCModel() {}

  std::string name() const { return model_->name(); }

  model::proto::Task task() const { return model_->task(); }

  absl::StatusOr<std::string> Describe(bool full_details,
                                       bool text_format) const;

  std::vector<int> input_features() const { return model_->input_features(); }

  int label_col_idx() const { return model_->label_col_idx(); }

  // Benchmark the inference speed of the model.
  absl::StatusOr<BenchmarkInferenceCCResult> Benchmark(
      const dataset::VerticalDataset& dataset, double benchmark_duration,
      double warmup_duration, int batch_size);

  // Gets an engine of the model. If the engine does not exist, create it.
  // This method is not thread safe.
  absl::StatusOr<const serving::FastEngine*> GetEngine();

  // Save the model to `directory`. Use `file_prefix` for all model files if
  // specified.
  absl::Status Save(std::string_view directory,
                    std::optional<std::string> file_prefix) const;

  // TODO: Allow passing the output array as a parameter to reduce heap
  // allocations.
  absl::StatusOr<py::array_t<float>> Predict(
      const dataset::VerticalDataset& dataset);

  absl::StatusOr<metric::proto::EvaluationResults> Evaluate(
      const dataset::VerticalDataset& dataset,
      const metric::proto::EvaluationOptions& options);

  absl::StatusOr<utils::model_analysis::proto::StandaloneAnalysisResult>
  Analyze(const dataset::VerticalDataset& dataset,
          const utils::model_analysis::proto::Options& options);

  absl::StatusOr<utils::model_analysis::proto::PredictionAnalysisResult>
  AnalyzePrediction(
      const dataset::VerticalDataset& example,
      const utils::model_analysis::proto::PredictionAnalysisOptions& options);

  const dataset::proto::DataSpecification& data_spec() const {
    return model_->data_spec();
  }

  void set_data_spec(const dataset::proto::DataSpecification& data_spec) {
    *model_->mutable_data_spec() = data_spec;
  }

  model::proto::Metadata metadata() const;

  void set_metadata(const model::proto::Metadata& metadata) {
    model_->mutable_metadata()->Import(metadata);
  }

  const std::optional<model::proto::HyperparametersOptimizerLogs>&
  hyperparameter_optimizer_logs() const {
    return model_->hyperparameter_optimizer_logs();
  }

  absl::flat_hash_map<std::string, model::proto::VariableImportanceSet>
  VariableImportances() const {
    return model_->precomputed_variable_importances();
  }

 protected:
  std::unique_ptr<model::AbstractModel> model_;
  utils::concurrency::Mutex engine_mutex_;
  std::unique_ptr<serving::FastEngine> engine_ GUARDED_BY(engine_mutex_);
};

}  // namespace yggdrasil_decision_forests::port::python

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_LEARNER_MODEL_H_
