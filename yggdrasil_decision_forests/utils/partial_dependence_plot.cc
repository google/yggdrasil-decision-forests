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

#include "yggdrasil_decision_forests/utils/partial_dependence_plot.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/example_reader.h"
#include "yggdrasil_decision_forests/dataset/example_reader_interface.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/partial_dependence_plot.pb.h"

namespace yggdrasil_decision_forests {
namespace utils {

using dataset::proto::ColumnType;
using dataset::proto::DataSpecification;
using dataset::proto::Example;
using model::proto::Prediction;
using proto::PartialDependencePlotSet;

namespace {

// Tells if a distribution would be better plotted in log scale, or uniform
// scale.
bool LogScaleBetterThanUniform(const std::vector<float>& bounds) {
  if (bounds.empty() || bounds.front() < 0) {
    return false;
  }
  const float margin = 0.10f;
  const float mid_value = bounds[bounds.size() / 2];
  return (mid_value - bounds.front()) <
         margin * (bounds.back() - bounds.front());
}

// Checks if a column type is supported.
bool SupportedFeatureType(const dataset::proto::ColumnType type) {
  // Not all feature types are supported.
  switch (type) {
    case dataset::proto::ColumnType::NUMERICAL:
    case dataset::proto::ColumnType::CATEGORICAL:
    case dataset::proto::ColumnType::BOOLEAN:
      return true;
    default:
      return false;
  }
}

// Extracts the input features of a model that are supported for PDP plotting.
std::vector<int> SupportedInputFeatures(const model::AbstractModel& model) {
  std::vector<int> supported_input_features;
  for (const auto col_idx : model.input_features()) {
    if (SupportedFeatureType(model.data_spec().columns(col_idx).type())) {
      supported_input_features.push_back(col_idx);
    }
  }
  return supported_input_features;
}

// Number of times the model needs to be called for each example when calling
// "UpdatePartialDependencePlotSet".
int64_t NumModelCallPerExample(const proto::PartialDependencePlotSet& pdp_set) {
  int64_t num_model_calls = 0;
  for (const auto& pdp : pdp_set.pdps()) {
    num_model_calls += pdp.pdp_bins_size();
  }
  return num_model_calls;
}

// Returns the index of the bin of a feature/attribute value.
absl::StatusOr<int> GetPerAttributeBinIdx(
    const Example& example, const dataset::proto::DataSpecification& data_spec,
    const proto::PartialDependencePlotSet::PartialDependencePlot::AttributeInfo&
        attribute_info) {
  const int attribute_idx = attribute_info.attribute_idx();
  const auto& attribute_spec = data_spec.columns(attribute_idx);
  switch (attribute_spec.type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      const auto min = attribute_spec.numerical().min_value();
      const auto max = attribute_spec.numerical().max_value();
      if (min == max) {
        return 0;
      }
      const float value = example.attributes(attribute_idx).numerical();
      const auto& boundaries = attribute_info.numerical_boundaries();
      const auto it =
          std::upper_bound(boundaries.begin(), boundaries.end(), value);
      const int idx = std::distance(boundaries.begin(), it);
      return utils::clamp(idx, 0,
                          attribute_info.num_bins_per_input_feature() - 1);
    }
    case dataset::proto::ColumnType::CATEGORICAL:
      return example.attributes(attribute_idx).categorical();
    case dataset::proto::ColumnType::BOOLEAN:
      return example.attributes(attribute_idx).boolean();
    default:
      return absl::InvalidArgumentError("Not supported attribute type");
  }
}

// Updates the fields necessary to the display of the attribute density in a pdp
// of a given example.
absl::Status UpdateDensity(
    const dataset::proto::DataSpecification& data_spec,
    const dataset::proto::Example& example, const float example_weight,
    proto::PartialDependencePlotSet::PartialDependencePlot* pdp) {
  for (int attribute_info_idx = 0;
       attribute_info_idx < pdp->attribute_info_size(); attribute_info_idx++) {
    ASSIGN_OR_RETURN(
        int per_attribute_bin_idx,
        GetPerAttributeBinIdx(example, data_spec,
                              pdp->attribute_info(attribute_info_idx)));
    auto& per_attribute_bin_num_obs =
        *pdp->mutable_attribute_info(attribute_info_idx)
             ->mutable_num_observations_per_bins();
    per_attribute_bin_num_obs.Set(
        per_attribute_bin_idx,
        per_attribute_bin_num_obs.Get(per_attribute_bin_idx) + example_weight);
  }
  return absl::OkStatus();
}

// Updates a PDP bin with the attributes from of an example.
absl::Status UpdateBin(
    const model::AbstractModel& model, const Prediction& prediction,
    const bool has_ground_truth,
    proto::PartialDependencePlotSet::PartialDependencePlot::Bin* bin) {
  switch (model.task()) {
    case model::proto::Task::CLASSIFICATION: {
      STATUS_CHECK(bin->prediction().has_classification_class_distribution());
      // Prediction.
      AddNormalizedToIntegerDistributionProto<proto::IntegerDistributionFloat>(
          prediction.classification().distribution(), prediction.weight(),
          bin->mutable_prediction()
              ->mutable_classification_class_distribution());
      if (has_ground_truth) {
        // Ground truth.
        AddToIntegerDistributionProto<proto::IntegerDistributionFloat>(
            prediction.classification().ground_truth(), prediction.weight(),
            bin->mutable_ground_truth()
                ->mutable_classification_class_distribution());
        // Evaluation
        if (prediction.classification().ground_truth() ==
            prediction.classification().value()) {
          bin->mutable_evaluation()->set_num_correct_predictions(
              bin->evaluation().num_correct_predictions() +
              prediction.weight());
        }
      }
    } break;

    case model::proto::Task::REGRESSION: {
      STATUS_CHECK(bin->prediction().has_sum_of_regression_predictions());
      // Prediction.
      bin->mutable_prediction()->set_sum_of_regression_predictions(
          bin->prediction().sum_of_regression_predictions() +
          prediction.regression().value() * prediction.weight());
      if (has_ground_truth) {
        // Ground truth.
        bin->mutable_ground_truth()->set_sum_of_regression_predictions(
            bin->ground_truth().sum_of_regression_predictions() +
            prediction.regression().ground_truth() * prediction.weight());
        // Evaluation
        const auto residual = prediction.regression().ground_truth() -
                              prediction.regression().value();
        bin->mutable_evaluation()->set_sum_squared_error(
            bin->evaluation().sum_squared_error() +
            prediction.weight() * residual * residual);
      }
    } break;

    case model::proto::Task::RANKING: {
      STATUS_CHECK(bin->prediction().has_sum_of_ranking_predictions());
      // Prediction.
      bin->mutable_prediction()->set_sum_of_ranking_predictions(
          bin->prediction().sum_of_ranking_predictions() +
          prediction.ranking().relevance() * prediction.weight());
      // TODO: Add direction of improvement. Unlike other tasks, the ground
      // truth does not have the same scale/range as the predictions.
    } break;

    default:
      return absl::InvalidArgumentError("Invalid model task");
  }
  return absl::OkStatus();
}

// Takes as input the set of possible values the attributes take
// (center_input_feature_values_vector), and a number less than \product_i
// center_input_feature_values_vector[i].size, to compute a set of indices in
// each attribute vector to store in the bin.
// For example let this correspond to three attributes , which take 2,3 and 4
// values.
// For example, if center_input_feature_values_vector = {{'En', 'De'}, {1, 2,
// 3}, {0.1, 0.3, 0.4, 0.5}}.
// Then, if index = 1, this stores {'De', 1, 0.1} in the bin.
// If index = 17, then the following steps are done:
//   17i % (vector[0].size()) = 17 % 2 = 1.
//   Hence vector[0][1] = 'De' is stored.
//   (17 - 1)/2 = 8.
//
//   8 % (vector[1].size()) = 8 % 3 = 2.
//   Hence vector[1][2] = 3 is stored.
//   (8 - 2)/3 = 2.
//
//   2 % (vector[2].size()) = 2 % 4 = 2.
//   Hence vector[2][2] = 0.4 is stored.
absl::Status IndexToBinCenter(
    const std::vector<std::vector<Example::Attribute>>&
        center_input_feature_values_vector,
    const int32_t index,
    PartialDependencePlotSet::PartialDependencePlot::Bin* bin) {
  STATUS_CHECK_GE(index, 0);
  int32_t number = index;
  for (int feature_idx = 0;
       feature_idx < center_input_feature_values_vector.size(); feature_idx++) {
    const int32_t remainder =
        number % center_input_feature_values_vector[feature_idx].size();
    *bin->add_center_input_feature_values() =
        center_input_feature_values_vector[feature_idx][remainder];
    number = (number - remainder) /
             center_input_feature_values_vector[feature_idx].size();
  }
  STATUS_CHECK_EQ(number, 0);
  return absl::OkStatus();
}

// Initializes the PDP by creating the required "pdp_bins" and storing the
// center values of each attribute in the bins.
absl::Status InitializePartialDependence(
    const DataSpecification& data_spec, const std::vector<int>& attribute_idxs,
    const model::proto::Task& task, const int label_col_idx,
    const int num_numerical_bins, const bool has_ground_truth,
    const dataset::VerticalDataset& dataset,
    PartialDependencePlotSet::PartialDependencePlot* pdp) {
  std::vector<std::vector<Example::Attribute>>
      center_input_feature_values_vector;
  int num_possible_values = 1;
  for (const int attribute_idx : attribute_idxs) {
    ASSIGN_OR_RETURN(const auto bins, internal::GetBinsForOneAttribute(
                                          data_spec, attribute_idx,
                                          num_numerical_bins, dataset));
    center_input_feature_values_vector.push_back(bins.centers);
    auto* attribute_info = pdp->add_attribute_info();
    const int num_bins_per_input_feature =
        center_input_feature_values_vector.back().size();
    attribute_info->set_num_bins_per_input_feature(num_bins_per_input_feature);
    attribute_info->set_attribute_idx(attribute_idx);
    *attribute_info->mutable_numerical_boundaries() = {
        bins.numerical_boundaries.begin(), bins.numerical_boundaries.end()};
    if (bins.is_log) {
      attribute_info->set_scale(
          PartialDependencePlotSet::PartialDependencePlot::AttributeInfo::LOG);
    }
    num_possible_values *= center_input_feature_values_vector.back().size();
    attribute_info->mutable_num_observations_per_bins()->Resize(
        num_bins_per_input_feature, 0);
  }

  for (int i = 0; i < num_possible_values; ++i) {
    auto* bin = pdp->add_pdp_bins();
    RETURN_IF_ERROR(
        IndexToBinCenter(center_input_feature_values_vector, i, bin));
    switch (task) {
      case model::proto::Task::CLASSIFICATION: {
        STATUS_CHECK_LT(label_col_idx, data_spec.columns_size());
        STATUS_CHECK(data_spec.columns(label_col_idx).has_categorical());
        const int num_classes = data_spec.columns(label_col_idx)
                                    .categorical()
                                    .number_of_unique_values();
        InitializeIntegerDistributionProto(
            num_classes, bin->mutable_prediction()
                             ->mutable_classification_class_distribution());
        if (has_ground_truth) {
          InitializeIntegerDistributionProto(
              num_classes, bin->mutable_ground_truth()
                               ->mutable_classification_class_distribution());
        }
      } break;
      case model::proto::Task::REGRESSION:
        bin->mutable_prediction()->set_sum_of_regression_predictions(0.0);
        break;

      case model::proto::Task::RANKING:
        bin->mutable_prediction()->set_sum_of_ranking_predictions(0.0);
        break;

      default:
        return absl::InvalidArgumentError("Invalid task");
    }
  }
  return absl::OkStatus();
}

// Sets the feature values of an example to be the center of the bins.
void ModifyExample(
    const PartialDependencePlotSet::PartialDependencePlot::Bin* bin,
    const PartialDependencePlotSet::PartialDependencePlot* pdp,
    Example* example) {
  for (int i = 0; i < pdp->attribute_info_size(); ++i) {
    const int attribute_idx = pdp->attribute_info(i).attribute_idx();
    *(example->mutable_attributes(attribute_idx)) =
        bin->center_input_feature_values(i);
  }
}

}  // namespace

absl::StatusOr<PartialDependencePlotSet> InitializePartialDependencePlotSet(
    const DataSpecification& data_spec,
    const std::vector<std::vector<int>>& attribute_idxs,
    const model::proto::Task& task, const int label_col_idx,
    const int num_numerical_bins, const dataset::VerticalDataset& dataset) {
  PartialDependencePlotSet pdp_set;
  for (const auto& set_of_attribute_idxs : attribute_idxs) {
    auto* pdp = pdp_set.add_pdps();
    RETURN_IF_ERROR(InitializePartialDependence(
        data_spec, set_of_attribute_idxs, task, label_col_idx,
        num_numerical_bins, false, dataset, pdp));
  }
  return pdp_set;
}

absl::StatusOr<ConditionalExpectationPlotSet>
InitializeConditionalExpectationPlotSet(
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<std::vector<int>>& attribute_idxs,
    const model::proto::Task& task, int label_col_idx, int num_numerical_bins,
    const dataset::VerticalDataset& dataset) {
  PartialDependencePlotSet pdp_set;
  for (const auto& set_of_attribute_idxs : attribute_idxs) {
    auto* pdp = pdp_set.add_pdps();
    RETURN_IF_ERROR(InitializePartialDependence(
        data_spec, set_of_attribute_idxs, task, label_col_idx,
        num_numerical_bins, true, dataset, pdp));
  }
  return pdp_set;
}

absl::Status UpdatePartialDependencePlotSet(const model::AbstractModel& model,
                                            const Example& example,
                                            PartialDependencePlotSet* pdp_set) {
  Prediction prediction;
  if (model.weights().has_value()) {
    prediction.set_weight(
        dataset::GetWeightWithStatus(example, model.weights().value()).value());
  }

  for (int pdp_idx = 0; pdp_idx < pdp_set->pdps_size(); ++pdp_idx) {
    auto* pdp = pdp_set->mutable_pdps(pdp_idx);
    // Density.
    RETURN_IF_ERROR(
        UpdateDensity(model.data_spec(), example, prediction.weight(), pdp));

    // PDP.
    Example modified_example = example;
    for (int bin_idx = 0; bin_idx < pdp->pdp_bins_size(); ++bin_idx) {
      auto* bin = pdp->mutable_pdp_bins(bin_idx);
      ModifyExample(bin, pdp, &modified_example);

      model.Predict(modified_example, &prediction);
      RETURN_IF_ERROR(UpdateBin(model, prediction, false, bin));
    }
    pdp->set_num_observations(pdp->num_observations() + prediction.weight());
  }
  return absl::OkStatus();
}

absl::Status UpdateConditionalExpectationPlotSet(
    const model::AbstractModel& model, const dataset::proto::Example& example,
    ConditionalExpectationPlotSet* cond_set) {
  // Apply the model.
  Prediction prediction;
  if (model.weights().has_value()) {
    prediction.set_weight(
        dataset::GetWeightWithStatus(example, model.weights().value()).value());
  }

  model.Predict(example, &prediction);
  RETURN_IF_ERROR(model.SetGroundTruth(example, &prediction));

  for (int cond_idx = 0; cond_idx < cond_set->pdps_size(); ++cond_idx) {
    auto* cond = cond_set->mutable_pdps(cond_idx);
    // Density.
    RETURN_IF_ERROR(
        UpdateDensity(model.data_spec(), example, prediction.weight(), cond));

    // Conditional expectation.
    ASSIGN_OR_RETURN(const int bin_idx, internal::ExampleToBinIndex(
                                            example, model.data_spec(), *cond));
    auto* bin = cond->mutable_pdp_bins(bin_idx);
    RETURN_IF_ERROR(UpdateBin(model, prediction, true, bin));

    cond->set_num_observations(cond->num_observations() + prediction.weight());
  }
  return absl::OkStatus();
}

absl::Status AppendAttributesCombinations(
    const model::AbstractModel& model, const int num_dims,
    std::vector<std::vector<int>>* attribute_idxs) {
  const auto supported_input_features = SupportedInputFeatures(model);
  if (num_dims == 1) {
    for (const auto col_idx : supported_input_features) {
      attribute_idxs->push_back({col_idx});
    }
  } else if (num_dims == 2) {
    const int n = supported_input_features.size();
    for (int idx_1 = 0; idx_1 < n; idx_1++) {
      for (int idx_2 = idx_1 + 1; idx_2 < n; idx_2++) {
        attribute_idxs->push_back(
            {supported_input_features[idx_1], supported_input_features[idx_2]});
      }
    }
  } else {
    return absl::InvalidArgumentError("Non supported num_dims");
  }
  return absl::OkStatus();
}

absl::Status AppendAttributesCombinations2D(
    const model::AbstractModel& model, const dataset::proto::ColumnType type_1,
    const dataset::proto::ColumnType type_2,
    std::vector<std::vector<int>>* attribute_idxs) {
  const auto supported_input_features = SupportedInputFeatures(model);
  for (const auto feature_1 : supported_input_features) {
    if (model.data_spec().columns(feature_1).type() != type_1) {
      continue;
    }
    for (const auto feature_2 : supported_input_features) {
      if (model.data_spec().columns(feature_2).type() != type_2) {
        continue;
      }
      if (type_1 == type_2 && feature_1 >= feature_2) {
        // If the two types are similar, we skip the attribute groups
        // <attr2,attr1> (Since <attr1,attr2> is already present).
        continue;
      }
      attribute_idxs->push_back({feature_1, feature_2});
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<proto::PartialDependencePlotSet> ComputePartialDependencePlotSet(
    const dataset::VerticalDataset& dataset, const model::AbstractModel& model,
    const std::vector<std::vector<int>>& attribute_idxs,
    const int num_numerical_bins, const float example_sampling) {
  YDF_LOG(INFO) << "Initiate PDP accumulator";
  ASSIGN_OR_RETURN(auto pdp_set,
                   InitializePartialDependencePlotSet(
                       model.data_spec(), attribute_idxs, model.task(),
                       model.label_col_idx(), num_numerical_bins, dataset));
  YDF_LOG(INFO) << "Compute partial dependence plot for "
                << attribute_idxs.size() << " set of features and "
                << NumModelCallPerExample(pdp_set)
                << " model call(s) per example.";

  std::default_random_engine random;
  std::uniform_real_distribution<float> dist_unif_unit;

  // TODO: Multi-thread.
  dataset::proto::Example example;
  for (size_t example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    if (example_sampling < 1.f && example_sampling < dist_unif_unit(random)) {
      continue;
    }
    if ((example_idx % 100) == 0) {
      LOG_INFO_EVERY_N_SEC(30, _ << example_idx + 1 << " examples scanned.");
    }
    dataset.ExtractExample(example_idx, &example);

    RETURN_IF_ERROR(UpdatePartialDependencePlotSet(model, example, &pdp_set));
  }

  return pdp_set;
}

absl::StatusOr<ConditionalExpectationPlotSet>
ComputeConditionalExpectationPlotSet(
    const dataset::VerticalDataset& dataset, const model::AbstractModel& model,
    const std::vector<std::vector<int>>& attribute_idxs, int num_numerical_bins,
    float example_sampling) {
  YDF_LOG(INFO) << "Initiate CEP accumulator";
  ASSIGN_OR_RETURN(auto pdp_set,
                   InitializeConditionalExpectationPlotSet(
                       model.data_spec(), attribute_idxs, model.task(),
                       model.label_col_idx(), num_numerical_bins, dataset));
  YDF_LOG(INFO) << "Compute conditional expectation plot for "
                << attribute_idxs.size() << " set of features and "
                << NumModelCallPerExample(pdp_set)
                << " model call(s) per example.";

  std::default_random_engine random;
  std::uniform_real_distribution<float> dist_unif_01;

  // TODO: Multi-thread.
  dataset::proto::Example example;
  for (size_t example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    if (example_sampling < 1.f && example_sampling < dist_unif_01(random)) {
      continue;
    }
    if ((example_idx % 100) == 0) {
      LOG_INFO_EVERY_N_SEC(30, _ << example_idx + 1 << " examples scanned.");
    }
    dataset.ExtractExample(example_idx, &example);
    RETURN_IF_ERROR(
        UpdateConditionalExpectationPlotSet(model, example, &pdp_set));
  }

  return pdp_set;
}

absl::StatusOr<std::vector<std::vector<int>>> GenerateAttributesCombinations(
    const model::AbstractModel& model, const bool flag_1d, const bool flag_2d,
    const bool flag_2d_categorical_numerical) {
  YDF_LOG(INFO) << "List plotting attribute combinations";
  std::vector<std::vector<int>> attribute_idxs;
  if (flag_1d) {
    RETURN_IF_ERROR(
        utils::AppendAttributesCombinations(model, 1, &attribute_idxs));
  }
  if (flag_2d) {
    RETURN_IF_ERROR(
        utils::AppendAttributesCombinations(model, 2, &attribute_idxs));
  }
  if (flag_2d_categorical_numerical) {
    RETURN_IF_ERROR(utils::AppendAttributesCombinations2D(
        model, dataset::proto::ColumnType::CATEGORICAL,
        dataset::proto::ColumnType::NUMERICAL, &attribute_idxs));
  }
  // Remove duplicates.
  std::sort(attribute_idxs.begin(), attribute_idxs.end());
  attribute_idxs.erase(
      std::unique(attribute_idxs.begin(), attribute_idxs.end()),
      attribute_idxs.end());

  YDF_LOG(INFO) << "Found " << attribute_idxs.size() << " combination(s)";
  return attribute_idxs;
}

namespace internal {

std::vector<std::pair<float, int>> SortedUniqueCounts(
    std::vector<float> values) {
  if (values.empty()) {
    return {};
  }
  std::sort(values.begin(), values.end(), [](const float a, const float b) {
    const bool nan_a = !(a == a);
    const bool nan_b = !(b == b);

    if (nan_b) {
      // Sort the nans at the top of the list.
      return !nan_a;
    }

    if (nan_a) {
      // Ensure there is no operations on nans.
      return false;
    }

    return a < b;
  });
  std::vector<std::pair<float, int>> result;

  float cur_value = 0;
  int cur_num = 0;

  for (const auto value : values) {
    if (std::isnan(value)) {
      continue;
    }
    if (value != cur_value) {
      if (cur_num > 0) {
        result.push_back({cur_value, cur_num});
      }
      cur_value = value;
      cur_num = 0;
    }
    cur_num++;
  }
  if (cur_num > 0) {
    result.push_back({cur_value, cur_num});
  }
  return result;
}

absl::StatusOr<int> ExampleToBinIndex(
    const dataset::proto::Example& example,
    const dataset::proto::DataSpecification& data_spec,
    const PartialDependencePlotSet::PartialDependencePlot& pdp) {
  int64_t index = 0;
  int64_t factor_accumulator = 1;
  for (const auto& attribute_info : pdp.attribute_info()) {
    ASSIGN_OR_RETURN(const auto per_attribute_idx,
                     GetPerAttributeBinIdx(example, data_spec, attribute_info));
    index += factor_accumulator * per_attribute_idx;
    factor_accumulator *= attribute_info.num_bins_per_input_feature();
  }
  return index;
}

absl::StatusOr<BinsDefinition> GetBinsForOneAttribute(
    const DataSpecification& data_spec, const int attribute_idx,
    const int num_numerical_bins, const dataset::VerticalDataset& dataset) {
  BinsDefinition result;
  STATUS_CHECK_GT(data_spec.columns_size(), attribute_idx);
  const auto& column = data_spec.columns(attribute_idx);
  if (column.type() == ColumnType::NUMERICAL) {
    STATUS_CHECK(column.has_numerical());

    ASSIGN_OR_RETURN(
        const auto column_data,
        dataset.ColumnWithCastWithStatus<
            dataset::VerticalDataset::NumericalColumn>(attribute_idx));

    const auto unique_values_and_counts_vector =
        SortedUniqueCounts(column_data->values());

    ASSIGN_OR_RETURN(auto bounds,
                     dataset::GenDiscretizedBoundaries(
                         unique_values_and_counts_vector, num_numerical_bins,
                         /*min_obs_in_bins=*/5, {}));
    if (bounds.empty()) {
      bounds.push_back(column.numerical().mean());
    }

    result.is_log = LogScaleBetterThanUniform(bounds);

    result.numerical_boundaries = {bounds.begin(), bounds.end()};
    {
      Example::Attribute attribute;
      attribute.set_numerical(
          (unique_values_and_counts_vector.front().first + bounds.front()) / 2);
      result.centers.push_back(attribute);
    }
    for (int i = 0; i < bounds.size() - 1; i++) {
      Example::Attribute attribute;
      attribute.set_numerical((bounds[i] + bounds[i + 1]) / 2);
      result.centers.push_back(attribute);
    }
    {
      Example::Attribute attribute;
      attribute.set_numerical(
          (unique_values_and_counts_vector.back().first + bounds.back()) / 2);
      result.centers.push_back(attribute);
    }

  } else if (data_spec.columns(attribute_idx).type() ==
             ColumnType::CATEGORICAL) {
    for (int value_idx = 0;
         value_idx < column.categorical().number_of_unique_values();
         ++value_idx) {
      Example::Attribute attribute;
      attribute.set_categorical(value_idx);
      result.centers.push_back(attribute);
    }
  } else if (data_spec.columns(attribute_idx).type() == ColumnType::BOOLEAN) {
    for (const auto value : {false, true}) {
      Example::Attribute attribute;
      attribute.set_boolean(value);
      result.centers.push_back(attribute);
    }
  } else {
    return absl::InvalidArgumentError(
        "PDP is only implemented for Numerical, Categorical, and Boolean "
        "features");
  }
  return result;
}

}  // namespace internal
}  // namespace utils
}  // namespace yggdrasil_decision_forests
