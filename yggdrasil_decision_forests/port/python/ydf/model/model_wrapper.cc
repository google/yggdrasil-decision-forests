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

#include "ydf/model/model_wrapper.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/describe.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/benchmark/inference.h"
#include "yggdrasil_decision_forests/utils/model_analysis.h"
#include "yggdrasil_decision_forests/utils/model_analysis.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests::port::python {

std::unique_ptr<GenericCCModel> CreateCCModel(
    std::unique_ptr<model::AbstractModel> model_ptr) {
  auto rf_model = RandomForestCCModel::Create(model_ptr);
  if (rf_model.ok()) {
    // `model_ptr` is now invalid.
    return std::move(rf_model.value());
  }
  auto gbt_model = GradientBoostedTreesCCModel::Create(model_ptr);
  if (gbt_model.ok()) {
    // `model_ptr` is now invalid.
    return std::move(gbt_model.value());
  }
  // `model_ptr` is still valid.
  return std::make_unique<GenericCCModel>(std::move(model_ptr));
}

absl::StatusOr<const serving::FastEngine*> GenericCCModel::GetEngine() {
  if (engine_ == nullptr) {
    // Note: Not thread safe.
    ASSIGN_OR_RETURN(engine_, model_->BuildFastEngine());
  }
  return engine_.get();
}

absl::StatusOr<py::array_t<float>> GenericCCModel::Predict(
    const dataset::VerticalDataset& dataset) {
  py::array_t<float, py::array::c_style | py::array::forcecast> predictions;
  static_assert(predictions.itemsize() == sizeof(float),
                "A C++ float should have the same size as a numpy float");

  ASSIGN_OR_RETURN(const auto* engine, GetEngine());

  // Convert the prediction to the expected format.
  const int64_t num_prediction_dimensions = engine->NumPredictionDimension();

  const auto& engine_features = engine->features();
  const int64_t total_num_examples = dataset.nrow();
  constexpr int64_t kMaxBatchSize = 100;
  const int64_t batch_size = std::min(kMaxBatchSize, total_num_examples);
  auto batch_of_examples = engine->AllocateExamples(batch_size);
  predictions.resize({total_num_examples * num_prediction_dimensions});

  auto unchecked_predictions = predictions.mutable_unchecked();

  const int64_t num_batches =
      (total_num_examples + batch_size - 1) / batch_size;
  std::vector<float> batch_of_predictions;
  for (int64_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    const int64_t begin_example_idx = batch_idx * batch_size;
    const int64_t end_example_idx =
        std::min(begin_example_idx + batch_size, total_num_examples);
    const int effective_batch_size = end_example_idx - begin_example_idx;
    RETURN_IF_ERROR(CopyVerticalDatasetToAbstractExampleSet(
        dataset, begin_example_idx, end_example_idx, engine_features,
        batch_of_examples.get()));
    engine->Predict(*batch_of_examples, effective_batch_size,
                    &batch_of_predictions);

    // Copy this batch to the numpy array.
    const int64_t np_array_begin =
        begin_example_idx * num_prediction_dimensions;
    std::memcpy(unchecked_predictions.mutable_data(np_array_begin),
                batch_of_predictions.data(),
                batch_of_predictions.size() * sizeof(float));
  }
  if (num_prediction_dimensions > 1) {
    predictions =
        predictions.reshape({total_num_examples, num_prediction_dimensions});
  }
  return predictions;
}

absl::StatusOr<metric::proto::EvaluationResults> GenericCCModel::Evaluate(
    const dataset::VerticalDataset& dataset,
    const metric::proto::EvaluationOptions& options) {
  ASSIGN_OR_RETURN(const auto* engine, GetEngine());
  utils::RandomEngine rnd;
  ASSIGN_OR_RETURN(const auto evaluation,
                   model_->EvaluateWithEngine(*engine, dataset, options, &rnd));
  return evaluation;
}

absl::StatusOr<utils::model_analysis::proto::StandaloneAnalysisResult>
GenericCCModel::Analyze(const dataset::VerticalDataset& dataset,
                        const utils::model_analysis::proto::Options& options) {
  ASSIGN_OR_RETURN(const auto analysis,
                   utils::model_analysis::Analyse(*model_, dataset, options));
  return utils::model_analysis::CreateStandaloneAnalysis(*model_, dataset, "",
                                                         "", analysis);
}

absl::Status GenericCCModel::Save(
    std::string_view directory,
    const std::optional<std::string> file_prefix) const {
  return model::SaveModel(directory, model_.get(), {file_prefix});
}

absl::StatusOr<py::array_t<int32_t>> DecisionForestCCModel::PredictLeaves(
    const dataset::VerticalDataset& dataset) {
  py::array_t<int32_t, py::array::c_style | py::array::forcecast> leaves;

  const size_t num_examples = dataset.nrow();
  const size_t num_trees = df_model_->num_trees();

  leaves.resize({num_examples, num_trees});
  auto unchecked_leaves = leaves.mutable_unchecked();
  for (size_t example_idx = 0; example_idx < num_examples; example_idx++) {
    auto dst = absl::MakeSpan(unchecked_leaves.mutable_data(example_idx, 0),
                              num_trees);
    RETURN_IF_ERROR(df_model_->PredictGetLeaves(dataset, example_idx, dst));
  }

  return leaves;
}

absl::StatusOr<py::array_t<float>> DecisionForestCCModel::Distance(
    const dataset::VerticalDataset& dataset1,
    const dataset::VerticalDataset& dataset2) {
  py::array_t<float, py::array::c_style | py::array::forcecast> distances;
  const size_t n1 = dataset1.nrow();
  const size_t n2 = dataset2.nrow();
  distances.resize({n1, n2});
  auto dst = absl::MakeSpan(distances.mutable_data(), n1 * n2);
  RETURN_IF_ERROR(df_model_->Distance(dataset1, dataset2, dst));
  return distances;
}

absl::StatusOr<std::string> GenericCCModel::Describe(
    const bool full_details, const bool text_format) const {
  if (text_format) {
    return model_->DescriptionAndStatistics(full_details);
  } else {
    return model::DescribeModelHtml(*model_, utils::GenUniqueId());
  }
}

// TODO: Pass utils::BenchmarkInferenceRunOptions directly.
absl::StatusOr<BenchmarkInferenceCCResult> GenericCCModel::Benchmark(
    const dataset::VerticalDataset& dataset, const double benchmark_duration,
    const double warmup_duration, const int batch_size) {
  std::vector<utils::BenchmarkInferenceResult> results;
  const utils::BenchmarkInterfaceTimingOptions timing_options = {
      /*.benchmark_duration =*/benchmark_duration,
      /*.warmup_duration =*/warmup_duration,
  };
  const utils::BenchmarkInferenceRunOptions options{/*.batch_size =*/batch_size,
                                                    /*.runs =*/std::nullopt,
                                                    /*.time =*/timing_options};

  // Run engines.
  ASSIGN_OR_RETURN(const auto engine, GetEngine());
  RETURN_IF_ERROR(
      utils::BenchmarkFastEngine(options, *engine, *model_, dataset, &results));
  if (results.empty()) {
    return absl::InternalError("No benchmark results.");
  }
  return BenchmarkInferenceCCResult(results[0]);
}

absl::StatusOr<std::unique_ptr<RandomForestCCModel>>
RandomForestCCModel::Create(std::unique_ptr<model::AbstractModel>& model_ptr) {
  auto* rf_model = dynamic_cast<YDFModel*>(model_ptr.get());
  if (rf_model == nullptr) {
    return absl::InvalidArgumentError(
        "This model is not a random forest model.");
  }
  // Both release and the unique_ptr constructor are noexcept.
  model_ptr.release();
  std::unique_ptr<YDFModel> new_model_ptr(rf_model);

  return std::make_unique<RandomForestCCModel>(std::move(new_model_ptr),
                                               rf_model);
}

absl::StatusOr<std::unique_ptr<GradientBoostedTreesCCModel>>
GradientBoostedTreesCCModel::Create(
    std::unique_ptr<model::AbstractModel>& model_ptr) {
  auto* gbt_model = dynamic_cast<YDFModel*>(model_ptr.get());
  if (gbt_model == nullptr) {
    return absl::InvalidArgumentError(
        "This model is not a gradient boosted trees model.");
  }
  // Both release and the unique_ptr constructor are noexcept.
  model_ptr.release();
  std::unique_ptr<YDFModel> new_model_ptr(gbt_model);

  return std::make_unique<GradientBoostedTreesCCModel>(std::move(new_model_ptr),
                                                       gbt_model);
}

std::string BenchmarkInferenceCCResult::ToString() const {
  return absl::StrFormat(
      "Inference time per example and per cpu core: %.3f us "
      "(microseconds)\nEstimated over %d runs over %.3f seconds.\n* Measured "
      "with the C++ serving API. Check model.to_cpp() for details.",
      duration_per_example * 1000000, num_runs, benchmark_duration);
}

}  // namespace yggdrasil_decision_forests::port::python
