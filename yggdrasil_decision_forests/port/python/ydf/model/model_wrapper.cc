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
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/describe.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/benchmark/inference.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/model_analysis.h"
#include "yggdrasil_decision_forests/utils/model_analysis.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/shap.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests::port::python {

absl::StatusOr<std::shared_ptr<const serving::FastEngine>>
GenericCCModel::GetEngine() {
  utils::concurrency::MutexLock lock(&engine_mutex_);
  if (engine_ == nullptr || invalidate_engine_) {
    RETURN_IF_ERROR(model_->Validate());
    ASSIGN_OR_RETURN(engine_, model_->BuildFastEngine(force_engine_name_));
    invalidate_engine_ = false;
  }
  return engine_;
}

absl::StatusOr<py::array_t<float>> GenericCCModel::Predict(
    const dataset::VerticalDataset& dataset, bool use_slow_engine,
    const int num_threads) {
  if (use_slow_engine) {
    return PredictWithSlowEngine(dataset, num_threads);
  } else {
    return PredictWithFastEngine(dataset, num_threads);
  }
}

absl::StatusOr<std::pair<std::unordered_map<std::string, py::array_t<float>>,
                         py::array_t<float>>>
GenericCCModel::PredictShap(const dataset::VerticalDataset& dataset,
                            int num_threads) {
  const size_t num_examples = dataset.nrow();

  // Result numpy values
  py::array_t<float, py::array::c_style | py::array::forcecast> initial_values;
  py::array_t<float, py::array::c_style | py::array::forcecast> shap_values;
  static_assert(initial_values.itemsize() == sizeof(float),
                "A C++ float should have the same size as a numpy float");
  static_assert(shap_values.itemsize() == sizeof(float),
                "A C++ float should have the same size as a numpy float");

  ASSIGN_OR_RETURN(const auto expected_shape, utils::shap::GetShape(*model_));
  initial_values.resize({expected_shape.num_outputs});
  shap_values.resize({num_examples, expected_shape.num_attributes,
                      expected_shape.num_outputs});

  // Fast accessors
  auto unchecked_initial_values = initial_values.mutable_unchecked();
  auto unchecked_shap_values = shap_values.mutable_unchecked();

  {
    py::gil_scoped_release release;

    struct Cache {
      dataset::proto::Example example;
      utils::shap::ExampleShapValues example_shapes;
    };

    const auto create_cache = [&](size_t thread_idx, size_t num_threads,
                                  size_t block_size) -> Cache {
      Cache cache;
      return cache;
    };

    const auto run = [&unchecked_shap_values, &unchecked_initial_values,
                      &dataset, this](size_t block_idx, size_t begin_item_idx,
                                      size_t end_item_idx,
                                      Cache* cache) -> absl::Status {
      for (auto example_idx = begin_item_idx; example_idx < end_item_idx;
           example_idx++) {
        // Compute shape on example
        dataset.ExtractExample(example_idx, &cache->example);
        RETURN_IF_ERROR(utils::shap::tree_shap(
            *model_, cache->example, &cache->example_shapes,
            /*compute_bias=*/block_idx == 0));

        // Save values in numpy array
        const auto& raw_values = cache->example_shapes.values();
        std::transform(raw_values.begin(), raw_values.end(),
                       unchecked_shap_values.mutable_data(example_idx, 0, 0),
                       [](double v) { return static_cast<float>(v); });

        if (block_idx == 0) {
          // Save the initial values
          const auto& raw_values = cache->example_shapes.bias();
          std::transform(raw_values.begin(), raw_values.end(),
                         unchecked_initial_values.mutable_data(),
                         [](double v) { return static_cast<float>(v); });
        }
      }
      return absl::OkStatus();
    };

    RETURN_IF_ERROR(utils::concurrency::ConcurrentForLoopWithWorker<Cache>(
        /*num_items=*/num_examples,
        /*max_num_threads=*/num_threads,
        /*min_block_size=*/100,   // At least 100 examples in a batch
        /*max_block_size=*/2000,  // No more than 2k examples in a batch
        create_cache, run));
  }

  // Remove the output dimension if there is only one output.
  if (expected_shape.num_outputs == 1) {
    initial_values = initial_values.reshape({});
  }

  // Slice the SHAP value of each attribute.
  std::unordered_map<std::string, py::array_t<float>> map_shaps;
  const auto& input_features = model_->input_features();
  const absl::flat_hash_set<int> input_features_set(input_features.begin(),
                                                    input_features.end());
  for (size_t attribute_idx = 0; attribute_idx < expected_shape.num_attributes;
       attribute_idx++) {
    if (!input_features_set.contains(attribute_idx)) {
      continue;
    }
    py::tuple slice_index(3);
    slice_index[0] = py::slice(0, num_examples, 1);
    slice_index[1] = attribute_idx;
    if (expected_shape.num_outputs == 1) {
      slice_index[2] = 0;
    } else {
      slice_index[2] = py::slice(0, expected_shape.num_outputs, 1);
    }
    py::array slice = shap_values[slice_index];
    map_shaps[model_->data_spec().columns(attribute_idx).name()] = slice;
  }
  return std::make_pair(std::move(map_shaps), std::move(initial_values));
}

absl::StatusOr<py::array_t<float>> GenericCCModel::PredictWithFastEngine(
    const dataset::VerticalDataset& dataset, const int num_threads) {
  py::array_t<float, py::array::c_style | py::array::forcecast> predictions;
  static_assert(predictions.itemsize() == sizeof(float),
                "A C++ float should have the same size as a numpy float");

  ASSIGN_OR_RETURN(const auto engine, GetEngine());

  // Convert the prediction to the expected format.
  const int64_t num_prediction_dimensions = engine->NumPredictionDimension();

  const auto& engine_features = engine->features();
  const int64_t total_num_examples = dataset.nrow();
  predictions.resize({total_num_examples * num_prediction_dimensions});

  auto unchecked_predictions = predictions.mutable_unchecked();

  {
    py::gil_scoped_release release;

    struct Cache {
      std::unique_ptr<serving::AbstractExampleSet> batch_of_examples;
      std::vector<float> batch_of_predictions;
    };

    const auto create_cache = [&](size_t thread_idx, size_t num_threads,
                                  size_t block_size) -> Cache {
      Cache cache;
      cache.batch_of_examples = engine->AllocateExamples(block_size);
      cache.batch_of_predictions.resize(block_size);
      return cache;
    };

    const auto run = [&, num_prediction_dimensions](
                         size_t block_idx, size_t begin_item_idx,
                         size_t end_item_idx, Cache* cache) -> absl::Status {
      const size_t effective_batch_size = end_item_idx - begin_item_idx;
      RETURN_IF_ERROR(CopyVerticalDatasetToAbstractExampleSet(
          dataset, begin_item_idx, end_item_idx, engine_features,
          cache->batch_of_examples.get()));
      engine->Predict(*cache->batch_of_examples, effective_batch_size,
                      &cache->batch_of_predictions);

      // Copy this batch to the numpy array.
      const int64_t np_array_begin = begin_item_idx * num_prediction_dimensions;
      std::memcpy(unchecked_predictions.mutable_data(np_array_begin),
                  cache->batch_of_predictions.data(),
                  cache->batch_of_predictions.size() * sizeof(float));

      return absl::OkStatus();
    };

    RETURN_IF_ERROR(utils::concurrency::ConcurrentForLoopWithWorker<Cache>(
        /*num_items=*/total_num_examples,
        /*max_num_threads=*/num_threads,
        /*min_block_size=*/100,    // At least 100 examples in a batch
        /*max_block_size=*/10000,  // No more than 10k examples in a batch
        create_cache, run));
  }

  if (num_prediction_dimensions > 1) {
    predictions =
        predictions.reshape({total_num_examples, num_prediction_dimensions});
  }
  return predictions;
}

absl::StatusOr<py::array_t<float>> GenericCCModel::PredictWithSlowEngine(
    const dataset::VerticalDataset& dataset, const int num_threads) {
  py::array_t<float, py::array::c_style | py::array::forcecast> predictions;
  static_assert(predictions.itemsize() == sizeof(float),
                "A C++ float should have the same size as a numpy float");
  const int64_t total_num_examples = dataset.nrow();
  int64_t num_prediction_dimensions;
  if (model_->task() == model::proto::Task::CLASSIFICATION) {
    num_prediction_dimensions =
        model_->LabelColumnSpec().categorical().number_of_unique_values() - 1;
    if (num_prediction_dimensions == 2) {
      num_prediction_dimensions = 1;
    }
  } else if (model_->task() == model::proto::Task::CATEGORICAL_UPLIFT ||
             model_->task() == model::proto::Task::NUMERICAL_UPLIFT) {
    num_prediction_dimensions = model_->data_spec()
                                    .columns(model_->uplift_treatment_col_idx())
                                    .categorical()
                                    .number_of_unique_values() -
                                2;
  } else {
    num_prediction_dimensions = 1;
  }
  predictions.resize({total_num_examples * num_prediction_dimensions});

  struct Cache {
    model::proto::Prediction prediction;
  };

  const auto create_cache = [&](size_t thread_idx, size_t num_threads,
                                size_t block_size) -> Cache { return {}; };

  const auto run = [&, num_prediction_dimensions](
                       size_t block_idx, size_t begin_item_idx,
                       size_t end_item_idx, Cache* cache) -> absl::Status {
    for (size_t example_idx = begin_item_idx; example_idx < end_item_idx;
         example_idx++) {
      model_->Predict(dataset, example_idx, &cache->prediction);
      auto float_prediction = absl::MakeSpan(
          predictions.mutable_data(example_idx * num_prediction_dimensions),
          num_prediction_dimensions);
      model::ProtoToFloatPrediction(cache->prediction, model_->task(),
                                    float_prediction);
    }

    return absl::OkStatus();
  };

  RETURN_IF_ERROR(utils::concurrency::ConcurrentForLoopWithWorker<Cache>(
      /*num_items=*/total_num_examples,
      /*max_num_threads=*/num_threads,
      /*min_block_size=*/100,    // At least 100 examples in a batch
      /*max_block_size=*/10000,  // No more than 10k examples in a batch
      create_cache, run));

  if (num_prediction_dimensions > 1) {
    predictions =
        predictions.reshape({total_num_examples, num_prediction_dimensions});
  }
  return predictions;
}

absl::StatusOr<metric::proto::EvaluationResults> GenericCCModel::Evaluate(
    const dataset::VerticalDataset& dataset,
    const metric::proto::EvaluationOptions& options, const bool weighted,
    const int label_col_idx, const int group_col_idx,
    const bool use_slow_engine, const int num_threads) {
  py::gil_scoped_release release;
  auto effective_options = options;
  if (weighted && model_->weights().has_value()) {
    ASSIGN_OR_RETURN(*effective_options.mutable_weights(),
                     dataset::GetUnlinkedWeightDefinition(
                         model_->weights().value(), model_->data_spec()));
  }
  if (use_slow_engine) {
    effective_options.set_force_slow_engine(true);
    utils::RandomEngine rnd;

    if (label_col_idx == model_->label_col_idx() &&
        group_col_idx == model_->ranking_group_col_idx() &&
        effective_options.task() == model_->task()) {
      // Model default evaluation
      return model_->Evaluate(dataset, effective_options, &rnd);
    } else {
      // Model evaluation with overrides
      return model_->EvaluateOverrideType(dataset, effective_options,
                                          effective_options.task(),
                                          label_col_idx, group_col_idx, &rnd);
    }
  } else {
    ASSIGN_OR_RETURN(const auto engine, GetEngine());
    utils::RandomEngine rnd;

    if (label_col_idx == model_->label_col_idx() &&
        group_col_idx == model_->ranking_group_col_idx() &&
        effective_options.task() == model_->task()) {
      // Model default evaluation
      return model_->EvaluateWithEngine(*engine, dataset, effective_options,
                                        &rnd);
    } else {
      // Model evaluation with overrides
      return model_->EvaluateWithEngineOverrideType(
          *engine, dataset, effective_options, effective_options.task(),
          label_col_idx, group_col_idx, &rnd);
    }
  }
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

absl::Status GenericCCModel::Save(std::string_view directory,
                                  const std::optional<std::string> file_prefix,
                                  const bool pure_serving) const {
  py::gil_scoped_release release;
  if (pure_serving) {
    // Do not modify the original model.
    ASSIGN_OR_RETURN(auto serialized_model, model::SerializeModel(*model_));
    ASSIGN_OR_RETURN(auto copied_model,
                     model::DeserializeModel(std::move(serialized_model)));
    RETURN_IF_ERROR(copied_model->MakePureServing());
    return model::SaveModel(directory, copied_model.get(), {file_prefix});
  } else {
    return model::SaveModel(directory, model_.get(), {file_prefix});
  }
}

absl::StatusOr<py::bytes> GenericCCModel::Serialize() const {
  ASSIGN_OR_RETURN(std::string serialized_model,
                   model::SerializeModel(*model_));
  return py::bytes(serialized_model);
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

absl::StatusOr<
    absl::flat_hash_map<std::string, model::proto::VariableImportanceSet>>
GenericCCModel::VariableImportances() const {
  RETURN_IF_ERROR(model_->PrecomputeVariableImportances(
      model_->AvailableVariableImportances()));
  return model_->precomputed_variable_importances();
}

// TODO: Pass utils::BenchmarkInferenceRunOptions directly.
absl::StatusOr<BenchmarkInferenceCCResult> GenericCCModel::Benchmark(
    const dataset::VerticalDataset& dataset, const double benchmark_duration,
    const double warmup_duration, const int batch_size, const int num_threads) {
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
  ASSIGN_OR_RETURN(const auto engine, GetEngine());
  RETURN_IF_ERROR(
      utils::BenchmarkFastEngine(options, *engine, *model_, dataset, &results));

  RETURN_IF_ERROR(utils::BenchmarkFastEngineMultiThreaded(
      options, *engine, *model_, dataset, num_threads, &results));

  if (results.empty()) {
    return absl::InternalError("No benchmark results.");
  }

  const auto& single_thread_result = results[0];
  const auto& multi_thread_result = results[1];

  return BenchmarkInferenceCCResult{
      .duration_per_example =
          absl::ToDoubleSeconds(single_thread_result.duration_per_example),
      .benchmark_duration =
          absl::ToDoubleSeconds(single_thread_result.duration_per_example),
      .num_runs = single_thread_result.num_runs,
      .duration_per_example_multithread =
          absl::ToDoubleSeconds(multi_thread_result.duration_per_example),
      .benchmark_duration_multithread =
          absl::ToDoubleSeconds(multi_thread_result.duration_per_example),
      .num_runs_multithread = multi_thread_result.num_runs,
      .num_threads = num_threads,
      .batch_size = single_thread_result.batch_size,
      .num_examples = static_cast<size_t>(dataset.nrow()),
  };
}

std::optional<int> GenericCCModel::weight_col_idx() const {
  if (!model_->weights()) {
    return std::nullopt;
  }
  return model_->weights()->attribute_idx();
}

std::string BenchmarkInferenceCCResult::ToString() const {
  return absl::StrFormat(
      R"BLOCK(Single-thread inference time per example: %.3f us (microseconds)
Details: %d predictions in %.3f seconds

Multi-thread inference time per example: %.3f us (microseconds)
Details: %d predictions in %.3f seconds using %d threads

* Measured with the C++ serving API. See model.to_cpp().)BLOCK",
      duration_per_example * 1000000, num_examples * num_runs,
      benchmark_duration, duration_per_example_multithread * 1000000,
      num_examples * num_runs_multithread, benchmark_duration_multithread,
      num_threads);
}

}  // namespace yggdrasil_decision_forests::port::python
