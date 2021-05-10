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

#include "yggdrasil_decision_forests/serving/utils.h"

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace {

using dataset::proto::ColumnType;

float OptionalOp(const float v1, const float v2, const bool has_v1,
                 const bool has_v2,
                 const std::function<float(const float v1, const float v2)> op,
                 const float default_value = 0.f) {
  if (has_v1 && !has_v2) {
    return v1;
  }
  if (has_v2 && !has_v1) {
    return v2;
  }
  if (!has_v1 && !has_v2) {
    return default_value;
  }
  return op(v1, v2);
}

}  // namespace

FeatureStatistics::FeatureStatistics(
    const dataset::proto::DataSpecification* data_spec,
    std::vector<int> feature_indices,
    std::vector<NumericalOrCategoricalValue> na_replacement_values)
    : data_spec_(data_spec),
      feature_indices_(std::move(feature_indices)),
      na_replacement_values_(std::move(na_replacement_values)) {
  for (int feature_idx = 0; feature_idx < feature_indices_.size();
       feature_idx++) {
    auto& feature = *statistics_.add_features();
    const int data_spec_feature_idx = feature_indices_[feature_idx];
    const auto& col_spec = data_spec_->columns(data_spec_feature_idx);

    // Initialize the accumulators.
    switch (col_spec.type()) {
      case dataset::proto::ColumnType::NUMERICAL:
        feature.mutable_numerical();
        break;
      case dataset::proto::ColumnType::CATEGORICAL:
        feature.mutable_categorical();
        break;
      default:
        LOG(WARNING) << "Type of feature \"" << col_spec.name()
                     << " is supported.";
    }
  }
}

void FeatureStatistics::Update(
    const std::vector<NumericalOrCategoricalValue>& examples,
    const int num_examples, const ExampleFormat format) {
  statistics_.set_num_examples(statistics_.num_examples() + num_examples);

  for (int feature_idx = 0; feature_idx < feature_indices_.size();
       feature_idx++) {
    // Description of the feature.
    const int data_spec_feature_idx = feature_indices_[feature_idx];
    const auto& col_spec = data_spec_->columns(data_spec_feature_idx);
    auto& feature_stats = *statistics_.mutable_features(feature_idx);

    for (int example_idx = 0; example_idx < num_examples; example_idx++) {
      // Get the feature value.
      int value_idx = 0;
      switch (format) {
        case ExampleFormat::FORMAT_FEATURE_MAJOR:
          value_idx = example_idx + feature_idx * num_examples;
          break;
        case ExampleFormat::FORMAT_EXAMPLE_MAJOR:
          value_idx = feature_idx + example_idx * feature_indices_.size();
          break;
        default:
          // Note: Warning was generated at initialization time.
          break;
      }
      const auto& value = examples[value_idx];

      // Is the feature missing?
      const bool is_missing = value == na_replacement_values_[feature_idx];
      if (!is_missing) {
        feature_stats.set_num_non_missing(feature_stats.num_non_missing() + 1);
      }

      // Update the accumulators.
      switch (col_spec.type()) {
        case dataset::proto::ColumnType::NUMERICAL: {
          if (is_missing) {
            continue;
          }
          auto& numerical = *feature_stats.mutable_numerical();
          if (feature_stats.num_non_missing() == 1) {
            numerical.set_min(value.numerical_value);
            numerical.set_max(value.numerical_value);
          } else {
            numerical.set_min(std::min(numerical.min(), value.numerical_value));
            numerical.set_max(std::max(numerical.max(), value.numerical_value));
          }
          numerical.set_sum(numerical.sum() + value.numerical_value);
          numerical.set_sum_squared(numerical.sum_squared() +
                                    value.numerical_value *
                                        value.numerical_value);
        } break;

        case dataset::proto::ColumnType::CATEGORICAL:
          (*feature_stats.mutable_categorical()
                ->mutable_count_per_value())[value.categorical_value]++;
          break;

        default:
          // Note: Warning was generated at initialization time.
          break;
      }
    }
  }
}

proto::FeatureStatistics FeatureStatistics::Export() const {
  return statistics_;
}

absl::Status FeatureStatistics::ImportAndAggregate(
    const proto::FeatureStatistics& src) {
  if (feature_indices_.size() != src.features_size()) {
    return absl::InvalidArgumentError(
        "The two FeatureStatistics are not compatible.");
  }
  return ImportAndAggregateProto(src, &statistics_);
}

absl::Status FeatureStatistics::ImportAndAggregateProto(
    const proto::FeatureStatistics& src, proto::FeatureStatistics* dst) {
  if (dst->num_examples() == 0) {
    *dst = src;
    return absl::OkStatus();
  }

  if (dst->features_size() != src.features_size()) {
    return absl::InvalidArgumentError(
        "The two FeatureStatistics are not compatible.");
  }

  dst->set_num_examples(dst->num_examples() + src.num_examples());

  for (int feature_idx = 0; feature_idx < src.features_size(); feature_idx++) {
    auto& src_feature = src.features(feature_idx);
    auto& dst_feature = *dst->mutable_features(feature_idx);

    switch (src_feature.type_case()) {
      case proto::FeatureStatistics::FeatureStatistic::kNumerical: {
        const auto& src_value = src_feature.numerical();
        auto& dst_value = *dst_feature.mutable_numerical();
        dst_value.set_sum(dst_value.sum() + src_value.sum());
        dst_value.set_sum_squared(dst_value.sum_squared() +
                                  src_value.sum_squared());
        dst_value.set_max(OptionalOp(
            src_value.max(), dst_value.max(), src_feature.num_non_missing() > 0,
            dst_feature.num_non_missing() > 0,
            [](const float v1, const float v2) { return std::max(v1, v2); }));
        dst_value.set_min(OptionalOp(
            src_value.min(), dst_value.min(), src_feature.num_non_missing() > 0,
            dst_feature.num_non_missing() > 0,
            [](const float v1, const float v2) { return std::min(v1, v2); }));
      } break;

      case proto::FeatureStatistics::FeatureStatistic::kCategorical: {
        auto& dst_items =
            *dst_feature.mutable_categorical()->mutable_count_per_value();
        for (const auto& src_item :
             src_feature.categorical().count_per_value()) {
          auto it_dst = dst_items.find(src_item.first);
          if (it_dst == dst_items.end()) {
            dst_items[src_item.first] = src_item.second;
          } else {
            it_dst->second += src_item.second;
          }
        }
      } break;

      default:
        // Note: Warning was generated at initialization time.
        break;
    }

    dst_feature.set_num_non_missing(dst_feature.num_non_missing() +
                                    src_feature.num_non_missing());
  }
  return absl::OkStatus();
}

std::string FeatureStatistics::BuildReport() const {
  std::string report;
  absl::StrAppend(&report, "FeatureStatistics report\n");
  absl::StrAppend(&report, "========================\n");

  absl::SubstituteAndAppend(&report, "Total number of features:$0\n",
                            data_spec_->columns_size());

  absl::SubstituteAndAppend(&report, "Model input features:$0\n",
                            feature_indices_.size());

  absl::SubstituteAndAppend(&report, "SERVING: Number of examples:$0\n",
                            statistics_.num_examples());
  absl::SubstituteAndAppend(&report, "MODEL: Number of examples:$0\n",
                            data_spec_->created_num_rows());

  absl::StrAppend(&report, "Features:\n");
  for (int feature_idx = 0; feature_idx < feature_indices_.size();
       feature_idx++) {
    const int data_spec_feature_idx = feature_indices_[feature_idx];
    const auto& col_spec = data_spec_->columns(data_spec_feature_idx);
    const auto& feature = statistics_.features(feature_idx);

    // Feature name, type and number of missing values.
    const auto num_missing_online =
        statistics_.num_examples() - feature.num_non_missing();
    const float ratio_missing_online =
        static_cast<float>(num_missing_online) / statistics_.num_examples();
    const auto num_missing_model = col_spec.count_nas();
    const float ratio_missing_model =
        static_cast<float>(num_missing_model) / data_spec_->created_num_rows();
    absl::StrAppendFormat(
        &report,
        "\t\"%s\" [%s] SERVING: num-missing:%d (%.2f%%) MODEL: "
        "num-missing:%d (%.2f%%)\n",
        col_spec.name(), dataset::proto::ColumnType_Name(col_spec.type()),
        num_missing_online, 100 * ratio_missing_online, num_missing_model,
        100 * ratio_missing_model);

    // Type dependent information.
    switch (col_spec.type()) {
      case dataset::proto::ColumnType::NUMERICAL: {
        // Show the mean, min, max, and sd of the feature.
        auto mean = std::numeric_limits<double>::quiet_NaN();
        auto sd = std::numeric_limits<double>::quiet_NaN();
        if (feature.num_non_missing() > 0) {
          mean = feature.numerical().sum() / feature.num_non_missing();
          sd = std::sqrt(feature.numerical().sum_squared() /
                             feature.num_non_missing() -
                         mean * mean);
        }

        absl::SubstituteAndAppend(
            &report, "\t\tSERVING: mean:$0 min:$1 max:$2 sd:$3\n", mean,
            feature.numerical().min(), feature.numerical().max(), sd);

        absl::SubstituteAndAppend(&report, "\t\tMODEL: mean:$0 min:$1 max:$2\n",
                                  col_spec.numerical().mean(),
                                  col_spec.numerical().min_value(),
                                  col_spec.numerical().max_value());
      } break;

      case dataset::proto::ColumnType::CATEGORICAL: {
        // Show the distribution of categorical values.

        // Join the distributions computed on the online and model datasets.
        struct Item {
          int value;
          int64_t count_online;
          int64_t count_model;
        };
        absl::flat_hash_map<int, Item> count_map;

        // Accumulate online statistics.
        for (const auto& item : feature.categorical().count_per_value()) {
          count_map[item.first].count_online = item.second;
        }
        // Accumulate model statistics.
        //
        // Note: Is "is_already_integerized", the dataspec does not contains the
        // distribution of values for the model dataset.
        //
        if (!col_spec.categorical().is_already_integerized()) {
          for (const auto& item : col_spec.categorical().items()) {
            count_map[item.second.index()].count_model = item.second.count();
          }
        }

        std::vector<Item> counts;
        counts.reserve(count_map.size());
        for (const auto& item : count_map) {
          counts.push_back({/*.value =*/item.first,
                            /*.count_online =*/item.second.count_online,
                            /*.count_model =*/item.second.count_model});
        }
        // Sort the item in decreasing frequency on the model dataset.
        std::sort(counts.begin(), counts.end(),
                  [](const Item& a, const Item& b) {
                    if (a.count_model != b.count_model) {
                      return a.count_model > b.count_model;
                    } else {
                      return a.value < b.value;
                    }
                  });

        for (const auto& item : counts) {
          absl::SubstituteAndAppend(
              &report,
              "\t\t\"$0\" SERVING: count:$1 ($2%) MODEL: count:$3 ($4%)\n",
              dataset::CategoricalIdxToRepresentation(col_spec, item.value),
              item.count_online,
              100.f * static_cast<float>(item.count_online) /
                  statistics_.num_examples(),
              item.count_model,
              100.f * static_cast<float>(item.count_model) /
                  data_spec_->created_num_rows());
        }
      } break;

      default:
        absl::SubstituteAndAppend(&report, "\t\tNon supported type");
    }
  }

  return report;
}

}  // namespace serving
}  // namespace yggdrasil_decision_forests
