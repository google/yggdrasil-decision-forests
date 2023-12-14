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
#include "absl/strings/str_cat.h"
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
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests::port::python {

absl::StatusOr<const serving::FastEngine*> GenericCCModel::GetEngine() {
  utils::concurrency::MutexLock lock(&engine_mutex_);
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

  {
    py::gil_scoped_release release;
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
  py::gil_scoped_release release;
  ASSIGN_OR_RETURN(const auto* engine, GetEngine());
  utils::RandomEngine rnd;
  ASSIGN_OR_RETURN(const auto evaluation,
                   model_->EvaluateWithEngine(*engine, dataset, options, &rnd));
  return evaluation;
}

absl::StatusOr<utils::model_analysis::proto::StandaloneAnalysisResult>
GenericCCModel::Analyze(const dataset::VerticalDataset& dataset,
                        const utils::model_analysis::proto::Options& options) {
  py::gil_scoped_release release;
  ASSIGN_OR_RETURN(const auto analysis,
                   utils::model_analysis::Analyse(*model_, dataset, options));
  return utils::model_analysis::CreateStandaloneAnalysis(*model_, dataset, "",
                                                         "", analysis);
}

absl::StatusOr<utils::model_analysis::proto::PredictionAnalysisResult>
GenericCCModel::AnalyzePrediction(
    const dataset::VerticalDataset& example,
    const utils::model_analysis::proto::PredictionAnalysisOptions& options) {
  py::gil_scoped_release release;
  if (example.nrow() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("The dataset should contain exactly one example. Instead "
                     "the dataset contains ",
                     example.nrow(), " example(s)"));
  }
  dataset::proto::Example example_proto;
  example.ExtractExample(0, &example_proto);
  return utils::model_analysis::AnalyzePrediction(*model_, example_proto,
                                                  options);
}

absl::Status GenericCCModel::Save(
    std::string_view directory,
    const std::optional<std::string> file_prefix) const {
  py::gil_scoped_release release;
  return model::SaveModel(directory, model_.get(), {file_prefix});
}

model::proto::Metadata GenericCCModel::metadata() const {
  model::proto::Metadata metadata;
  model_->metadata().Export(&metadata);
  return metadata;
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
  py::gil_scoped_release release;
  std::vector<utils::BenchmarkInferenceResult> results;
  const utils::BenchmarkInterfaceTimingOptions timing_options = {
      /*.benchmark_duration =*/benchmark_duration,
      /*.warmup_duration =*/warmup_duration,
  };
  const utils::BenchmarkInferenceRunOptions options{/*.batch_size =*/batch_size,
                                                    /*.runs =*/std::nullopt,
                                                    /*.time =*/timing_options};

  // Run engines.
  ASSIGN_OR_RETURN(const auto* engine, GetEngine());
  RETURN_IF_ERROR(
      utils::BenchmarkFastEngine(options, *engine, *model_, dataset, &results));
  if (results.empty()) {
    return absl::InternalError("No benchmark results.");
  }
  return BenchmarkInferenceCCResult(results[0]);
}

std::string BenchmarkInferenceCCResult::ToString() const {
  return absl::StrFormat(
      "Inference time per example and per cpu core: %.3f us "
      "(microseconds)\nEstimated over %d runs over %.3f seconds.\n* Measured "
      "with the C++ serving API. Check model.to_cpp() for details.",
      duration_per_example * 1000000, num_runs, benchmark_duration);
}

}  // namespace yggdrasil_decision_forests::port::python
