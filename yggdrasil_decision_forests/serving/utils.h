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

// Utility function for the serving of decision forests.
//
#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_UTILS_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_UTILS_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/serving/serving.pb.h"

namespace yggdrasil_decision_forests {
namespace serving {

// Computes and accumulates statistics on the features from a stream of
// examples. The statistics can then be printed and compared to the statistics
// contained in  the model and computed on the training dataset.
//
// This class is thread compatible.
//
// Usage example:
//
//  // Initialization.
//  CHECK_OK(GenericToSpecializedModel(*model, &fast_model));
//  FeatureStatistics stats(&model->data_spec(),
//                          fast_model.input_features_idxs,
//                          fast_model.na_replacement_values);
//
//  // For each new batch of examples.
//  std::vector<NumericalOrCategoricalValue> examples = ... ;
//  int num_examples = ...
//  stats.Update(examples, num_examples, ExampleFormat::FORMAT_EXAMPLE_MAJOR);
//
//  // Final display.
//  LOG(INFO) << "Statistics:\n", << stats.BuildReport();
//
// FutureWork(gbm): Create a method exporting results as a CSV file.
class FeatureStatistics {
 public:
  // Initialize the accumulator.
  //
  // Args:
  //   data_spec: NON-OWNING pointer to the dataspec describing the dataset. The
  //     pointed "DataSpecification" object should remain valid for the entire
  //     life of the "FeatureStatistics" object.
  //     Can be obtained with model.data_spec(). feature_indices: Indices of the
  //     features (in the dataspec) used by the model. Can be obtained with
  //     model.input_features().
  //   na_replacement_values: Representation of the missing value. Depend on the
  //     optimized model.
  FeatureStatistics(
      const dataset::proto::DataSpecification* data_spec,
      std::vector<int> feature_indices,
      std::vector<NumericalOrCategoricalValue> na_replacement_values);

  template <typename Model>
  std::vector<int> ExtractIndices(const Model& model) {
    std::vector<int> feature_indices;
    feature_indices.reserve(model.features().fixed_length_features().size());
    for (const auto& feature : model.features().fixed_length_features()) {
      feature_indices.push_back(feature.spec_idx);
    }
    return feature_indices;
  }

  // Initialize the feature statistics using the fast model api v2
  //
  // The "model" should outlive the "FeatureStatistics" object.
  template <typename Model>
  explicit FeatureStatistics(const Model& model)
      : FeatureStatistics(
            &model.features().data_spec(), ExtractIndices(model),
            model.features().fixed_length_na_replacement_values()) {}

  // Update the statistics from a new batch of examples.
  void Update(const std::vector<NumericalOrCategoricalValue>& examples,
              int num_examples, ExampleFormat format);

  // Update the statistics from a new batch of examples using the model api v2.
  template <typename Model>
  void Update(const typename Model::ExampleSet& examples, int num_examples,
              const Model& model);

  // Exports the content of the accumulator for later import using
  // "ImportAndAggregate".
  proto::FeatureStatistics Export() const;

  // Imports the statistics and aggregate the statistics. Can be used to merge
  // the work done by separate flume workers.
  absl::Status ImportAndAggregate(const proto::FeatureStatistics& src);

  // Generates a human readable report about the statistics.
  std::string BuildReport() const;

  // Similar to "ImportAndAggregate", but operates directly on the protos.
  static absl::Status ImportAndAggregateProto(
      const proto::FeatureStatistics& src, proto::FeatureStatistics* dst);

 private:
  const dataset::proto::DataSpecification* data_spec_;

  // List of features that will be provided at each update.
  const std::vector<int> feature_indices_;

  // Representation of a missing value.
  const std::vector<NumericalOrCategoricalValue> na_replacement_values_;

  // Accumulated statistics.
  proto::FeatureStatistics statistics_;
};

// =======================================
//   Below are the template definitions.
// =======================================

template <typename Model>
void FeatureStatistics::Update(const typename Model::ExampleSet& examples,
                               int num_examples, const Model& model) {
  Update(examples.InternalCategoricalAndNumericalValues(), num_examples,
         Model::ExampleSet::kFormat);
}

}  // namespace serving
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_UTILS_H_
