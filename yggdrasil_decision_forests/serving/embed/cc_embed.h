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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_CC_EMBED_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_CC_EMBED_H_

#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

// C++ headers to include in the generated source code.
struct Includes {
  // <array> include.
  bool array = false;
  // <algorithm> include.
  bool algorithm = false;
  // <cmath> include.
  bool cmath = false;
};

// Specific options for the generation of the model.
// The internal options contains all the precise internal decision aspect of the
// model compilation e.g. how many bits to use to encode numerical features. The
// internal options are computed using the user provided options (simply called
// "options" in the code) and the model.
struct InternalOptions {
  // Number of bytes to encode a fixed-size feature.
  // Note: Currently, all the fixed-size features are encoded with the same
  // precision (e.g. all the numerical and categorical values are encoded with
  // the same number of bytes). Can be 1, 2, or 4.
  int feature_value_bytes = 0;

  // If the numerical features are encoded as float. In this case
  // feature_value_bytes=4 (currently). If false, numerical features are encoded
  // as ints, and "feature_value_bytes" specify the precision.
  bool numerical_feature_is_float = false;

  // Number of bytes to encode a feature index.
  int feature_index_bytes = 0;

  // Number of bytes to encode a tree index.
  int tree_index_bytes;

  // Number of bytes to encode a node index.
  int node_index_bytes;

  // Number of bytes to encode a node index withing a tree.
  int node_offset_bytes;

  // Number of bytes to encode an index in the categorical mask bank.
  // Note: This value is currently inferred from
  // "sum_size_categorical_bitmap_masks", which assume the bank is not
  // compressed / optimized in any way.
  int categorical_idx_bytes;

  // The type returned by the prediction function.
  std::string output_type;

  // C++ includes.
  Includes includes;

  // Mapping from a column idx to a dense index of the model input features. If
  // a column is not a feature, the corresponding value is -1.
  std::vector<int> column_idx_to_feature_idx;

  // Mapping between column idx of a categorical-string column, to the sanitized
  // dictionary of possible values.
  struct CategoricalDict {
    // Name of the column
    std::string sanitized_name;
    // Possible values sanitized so they can be used as c++ variable names.
    std::vector<std::string> sanitized_items;
    // Possible values
    std::vector<std::string> items;
    // If this column a label.
    bool is_label;
  };
  absl::btree_map<int, CategoricalDict> categorical_dicts;
};

// Reserved feature index used for oblique conditions.
int ObliqueFeatureIndex(const InternalOptions& internal_options);

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelCC(
    const model::AbstractModel& model, const proto::Options& options);
// Type used to encode an oblique index.
// This cannot be computed with the other internal options as it depends on the
// forest node tracing.
std::string ObliqueFeatureType(const ValueBank& bank);

absl::StatusOr<SpecializedConversion> SpecializedConversionRandomForest(
    const model::random_forest::RandomForestModel& model,
    const internal::ModelStatistics& stats,
    const internal::InternalOptions& internal_options,
    const proto::Options& options);

absl::StatusOr<SpecializedConversion> SpecializedConversionGradientBoostedTrees(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& model,
    const internal::ModelStatistics& stats,
    const internal::InternalOptions& internal_options,
    const proto::Options& options);

// Computes the internal options of the model.
absl::StatusOr<InternalOptions> ComputeInternalOptions(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface,
    const ModelStatistics& stats, const proto::Options& options);

// Populates the feature parts of the internal option.
absl::Status ComputeInternalOptionsFeature(const ModelStatistics& stats,
                                           const model::AbstractModel& model,
                                           const proto::Options& options,
                                           InternalOptions* out);

// Populates the output parts of the internal option.
absl::Status ComputeInternalOptionsOutput(const ModelStatistics& stats,
                                          const proto::Options& options,
                                          InternalOptions* out);

// Populates the categorical dictionary parts of the internal option.
absl::Status ComputeInternalOptionsCategoricalDictionaries(
    const model::AbstractModel& model, const ModelStatistics& stats,
    const proto::Options& options, InternalOptions* out);

struct FeatureDef {
  std::string variable_name;  // Name used for the feature in the export.
  std::string type;  // Type to encode a feature using typedef / enum class.
  std::string underlying_type;  // Type to encode a feature e.g. "float".
  absl::optional<std::string> default_value = {};  // Optional default value.
  absl::optional<std::string> na_replacement =
      {};  // NA Replacement value for numerical features.
};

// Generates the definition of a feature in an instance struct.
absl::StatusOr<FeatureDef> GenFeatureDef(
    const dataset::proto::Column& col,
    const internal::InternalOptions& internal_options);

// Adds the code of a condition for the routing algorithm.
// If the model supports multiple types of condition, wrapps the code with the
// necessary branching.
absl::Status AddRoutingConditions(std::vector<RoutingConditionCode> conditions,
                                  const ValueBank& bank, std::string* content);

absl::Status CorePredict(const dataset::proto::DataSpecification& dataspec,
                         const model::DecisionForestInterface& df_interface,
                         const SpecializedConversion& specialized_conversion,
                         const ModelStatistics& stats,
                         const InternalOptions& internal_options,
                         const proto::Options& options,
                         const std::vector<FeatureDef>& feature_defs,
                         const ValueBank& routing_bank,
                         std::string* content_with_nan_replacement,
                         std::string* content_without_nan_replacement);

// The scalar type of an accumulator.
struct AccumulatorDef {
  std::string type;
  std::string base_type;
  bool use_array = false;
};
AccumulatorDef GenAccumulatorDef(const proto::Options& options,
                                 const ModelStatistics& stats);

absl::Status GenerateTreeInferenceIfElse(
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const proto::Options& options, const InternalOptions& internal_options,
    const IfElseSetNodeFn& set_node_ifelse_fn, std::string* content);

absl::Status GenerateTreeInferenceRouting(
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const proto::Options& options, const InternalOptions& internal_options,
    const SpecializedConversion& specialized_conversion,
    const ModelStatistics& stats, const ValueBank& routing_bank,
    std::string* content);

absl::Status GenRoutingModelData(
    const model::AbstractModel& model,
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const ModelStatistics& stats,
    const SpecializedConversion& specialized_conversion,
    const proto::Options& options, const InternalOptions& internal_options,
    std::string* content, ValueBank* bank);

}  // namespace yggdrasil_decision_forests::serving::embed::internal

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_CC_EMBED_H_
