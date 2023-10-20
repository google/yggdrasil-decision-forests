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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_LEARNER_MODEL_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_LEARNER_MODEL_H_

#include <pybind11/numpy.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/benchmark/inference.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/model_analysis.pb.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

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

  std::string Describe(const bool full_details) const {
    return model_->DescriptionAndStatistics(full_details);
  }

  std::vector<int> input_features() const { return model_->input_features(); }

  // Benchmark the inference speed of the model.
  absl::StatusOr<BenchmarkInferenceCCResult> Benchmark(
      const dataset::VerticalDataset& dataset, const double benchmark_duration,
      const double warmup_duration, const int batch_size);

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

  const dataset::proto::DataSpecification& data_spec() const {
    return model_->data_spec();
  }

  const std::optional<model::proto::HyperparametersOptimizerLogs>&
  hyperparameter_optimizer_logs() const {
    return model_->hyperparameter_optimizer_logs();
  }

 protected:
  std::unique_ptr<model::AbstractModel> model_;
  std::unique_ptr<serving::FastEngine> engine_;
};

class DecisionForestCCModel : public GenericCCModel {
 public:
  int num_trees() const { return df_model_->num_trees(); }

  absl::StatusOr<py::array_t<int32_t>> PredictLeaves(
      const dataset::VerticalDataset& dataset);

 protected:
  // `model` and `df_model` must correspond to the same object.
  DecisionForestCCModel(std::unique_ptr<model::AbstractModel>&& model,
                        model::DecisionForestInterface* df_model)
      : GenericCCModel(std::move(model)), df_model_(df_model) {}

 private:
  // This is a non-owning pointer to the model held by `model_`.
  model::DecisionForestInterface* df_model_;
};

class RandomForestCCModel : public DecisionForestCCModel {
  using YDFModel =
      ::yggdrasil_decision_forests::model::random_forest::RandomForestModel;

 public:
  // Creates a RandomForestCCModel if `model_ptr` refers to a RandomForestModel.
  //
  // If this method returns an invalid status, "model_ptr" is not modified.
  // If this method returns an ok status, the content of "model_ptr" is moved
  // (and "model_ptr" becomes empty).
  static absl::StatusOr<std::unique_ptr<RandomForestCCModel>> Create(
      std::unique_ptr<model::AbstractModel>& model_ptr);

  // `model` and `rf_model` must point to the same object. Prefer using
  // RandomForestCCModel::Compute for construction.
  RandomForestCCModel(std::unique_ptr<YDFModel> model, YDFModel* rf_model)
      : DecisionForestCCModel(std::move(model), rf_model), rf_model_(rf_model) {
    DCHECK_EQ(model_.get(), rf_model_);
  }

 private:
  // This is a non-owning pointer to the model held by `model_`.
  YDFModel* rf_model_;
};

class GradientBoostedTreesCCModel : public DecisionForestCCModel {
  using YDFModel = ::yggdrasil_decision_forests::model::gradient_boosted_trees::
      GradientBoostedTreesModel;

 public:
  // Creates a GradientBoostedTreesCCModel if `model_ptr` refers to a
  // GradientBoostedTreesModel.
  //
  // If this method returns an invalid status, "model_ptr" is not modified.
  // If this method returns an ok status, the content of "model_ptr" is
  // moved (and "model_ptr" becomes empty).
  static absl::StatusOr<std::unique_ptr<GradientBoostedTreesCCModel>> Create(
      std::unique_ptr<model::AbstractModel>& model_ptr);

  // `model` and `rf_model` must point to the same object. Prefer using
  // GradientBoostedTreesCCModel::Compute for construction.
  GradientBoostedTreesCCModel(std::unique_ptr<YDFModel> model,
                              YDFModel* gbt_model)
      : DecisionForestCCModel(std::move(model), gbt_model),
        gbt_model_(gbt_model) {
    DCHECK_EQ(model_.get(), gbt_model_);
  }

  // Return's the model's validation loss.
  float validation_loss() const { return gbt_model_->validation_loss(); }

 private:
  // This is a non-owning pointer to the model held by `model_`.
  YDFModel* gbt_model_;
};

std::unique_ptr<GenericCCModel> CreateCCModel(
    std::unique_ptr<model::AbstractModel> model_ptr);

}  // namespace yggdrasil_decision_forests::port::python

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_LEARNER_MODEL_H_
