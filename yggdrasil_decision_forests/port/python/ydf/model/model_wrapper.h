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

#include <atomic>
#include <cstddef>
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
  // Single thread
  double duration_per_example;
  double benchmark_duration;
  int num_runs;

  // Multi-thread
  double duration_per_example_multithread;
  double benchmark_duration_multithread;
  int num_runs_multithread;
  int num_threads;

  // Common
  int batch_size;
  size_t num_examples;

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

  int group_col_idx() const { return model_->ranking_group_col_idx(); }

  // Benchmark the inference speed of the model.
  absl::StatusOr<BenchmarkInferenceCCResult> Benchmark(
      const dataset::VerticalDataset& dataset, double benchmark_duration,
      double warmup_duration, int batch_size, int num_threads);

  // Gets an engine of the model. If the engine does not exist, create it.
  // This method is not thread safe.
  absl::StatusOr<std::shared_ptr<const serving::FastEngine>> GetEngine();

  // Save the model to `directory`. Use `file_prefix` for all model files if
  // specified.
  absl::Status Save(std::string_view directory,
                    std::optional<std::string> file_prefix,
                    bool pure_serving) const;

  // Serializes a model to a sequence of bytes.
  absl::StatusOr<py::bytes> Serialize() const;

  // TODO: Allow passing the output array as a parameter to reduce heap
  // allocations.
  absl::StatusOr<py::array_t<float>> Predict(
      const dataset::VerticalDataset& dataset, bool use_slow_engine,
      int num_threads);

  // Predict using one of the fast engines. This function is used in the vast
  // majority of cases.
  absl::StatusOr<py::array_t<float>> PredictWithFastEngine(
      const dataset::VerticalDataset& dataset, int num_threads);

  // Predict using the slow engine. This function should only be used for
  // debugging or edge cases.
  absl::StatusOr<py::array_t<float>> PredictWithSlowEngine(
      const dataset::VerticalDataset& dataset, int num_threads);

  absl::StatusOr<metric::proto::EvaluationResults> Evaluate(
      const dataset::VerticalDataset& dataset,
      const metric::proto::EvaluationOptions& options, bool weighted,
      int label_col_idx, int group_col_idx, bool use_slow_engine,
      int num_threads);

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

  void invalidate_engine() { invalidate_engine_ = true; }

  void ForceEngine(std::optional<std::string> engine_name) {
    // TODO: Let the user configure inference without an engine.
    utils::concurrency::MutexLock lock(&engine_mutex_);
    force_engine_name_ = engine_name;
    invalidate_engine();
  }

  std::vector<std::string> ListCompatibleEngines() const {
    return model_->ListCompatibleFastEngineNames();
  }

  // TODO: Remove when solved.
  bool weighted_training() const { return model_->weights().has_value(); }

  std::optional<model::proto::FeatureSelectionLogs> feature_selection_logs()
      const {
    return model_->feature_selection_logs();
  }

  void set_feature_selection_logs(
      std::optional<model::proto::FeatureSelectionLogs> value) {
    *model_->mutable_feature_selection_logs() = value;
  }

 protected:
  std::unique_ptr<model::AbstractModel> model_;
  utils::concurrency::Mutex engine_mutex_;
  std::shared_ptr<const serving::FastEngine> engine_ GUARDED_BY(engine_mutex_);

  // If true, the "engine_" is outdated (e.g., the model was modified) and
  // should be re-computed.
  std::atomic_bool invalidate_engine_{false};

  // If set, for the creation of this specific engine. If non set, fastest
  // compatible engine is created.
  std::optional<std::string> force_engine_name_;
};

}  // namespace yggdrasil_decision_forests::port::python

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_LEARNER_MODEL_H_
