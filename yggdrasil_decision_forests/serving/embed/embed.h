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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_EMBED_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_EMBED_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/structure_analysis.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"

namespace yggdrasil_decision_forests::serving::embed {

// Embed a model into a C++ library without dependencies to the YDF library.
// Returns a list of filenames and matching content.
typedef std::string Filename;
typedef std::string Content;
absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelCC(
    const model::AbstractModel& model, const proto::Options& options = {});

namespace internal {

// Statistics about the model.
struct ModelStatistics {
  // Number of trees.
  int64_t num_trees = 0;

  // Number of leaves.
  int64_t num_leaves = 0;

  // Number of conditions.
  int64_t num_conditions = 0;

  // Maximum number of leaves among all the trees.
  int64_t max_num_leaves_per_tree = 0;

  // Maximum depth. max_depth=0 indicates that the tree contains a single node.
  int64_t max_depth = 0;

  // Number of input features.
  int num_features = 0;

  // Task solved by the model.
  model::proto::Task task = model::proto::Task::UNDEFINED;

  // Number of label classes in the case of classification.
  int num_classification_classes = -1;

  // Sum of the sizes of all bitmap masks for categorical conditions.
  // Note: If the categorical mask bank is compressed, this value is smaller.
  int sum_size_categorical_bitmap_masks = 0;

  // Which conditions are used by the model.
  std::array<bool, model::decision_tree::kNumConditionTypes + 1> has_conditions{
      false};

  // True if "has_conditions" contains more than one true value i.e. the model
  // has more than one type of condition.
  bool has_multiple_condition_types;

  bool is_classification() const {
    return task == model::proto::Task::CLASSIFICATION;
  }

  bool is_binary_classification() const {
    return is_classification() && num_classification_classes == 2;
  }
};

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
    // Possible values
    std::vector<std::string> sanitized_items;
    // If this column a label.
    bool is_label;
  };
  absl::btree_map<int, CategoricalDict> categorical_dicts;
};

// Number of feature indexes to reserve.
// The precision of feature indexes should be sufficient to encode the index of
// any features + kReservedFeatureIndexes.
// For example, if the model has 254 features, but kReservedFeatureIndexes=2,
// the maximum feature index is 254+2=256, and feature indexes are encoded as
// uint16 (instead of uint8).
static constexpr int kReservedFeatureIndexes = 1;

// Index of the condition types supported by the routing algorithm.
enum class RoutingConditionType {
  HIGHER_CONDITION = 0,
  CONTAINS_CONDITION_BUFFER_BITMAP = 1,
  OBLIQUE_CONDITION = 2,
  NUM_ROUTING_CONDITION_TYPES = 3,
};

// Data that is not stored inside of the nodes with the routing algorithm.
struct ValueBank {
  // Values for categorical conditions.
  std::vector<bool> categorical;

  // Values of leaf nodes.
  std::vector<float> leaf_value;

  // The "root_deltas" contains the number of nodes in each tree. The node index
  // of the root of each tree can be computed by running a cumulative sum.
  std::vector<int> root_deltas;

  // Weights to encode oblique conditions. Also contain the threshold.
  //
  // An oblique condition for a node with a given "obl(ique")" is defined as
  // follow:
  // num_oblique_projections := oblique_features[obl]
  // threshold := oblique_weights[obl]
  // acc := 0
  // for i in 0..num_oblique_projections:
  //   offset := obl + i + 1
  //   acc += feature_value[oblique_features[offset]] *
  //     oblique_weights[offset];
  // eval := acc >= threshold
  std::vector<float> oblique_weights;

  // Feature index to encode oblique conditions. Also contain the number of
  // features in an oblique split. See definition of "oblique_weights".
  std::vector<size_t> oblique_features;

  // Number of conditions for each condition implementations.
  std::array<int, static_cast<int>(
                      RoutingConditionType::NUM_ROUTING_CONDITION_TYPES)>
      num_conditions{0};
};

// Code to evaluate a condition in the routing algorithm.
struct RoutingConditionCode {
  // Condition type. Used to determine if the code is needed for the model.
  // Also, used to define "cond" if not provided by the user.
  RoutingConditionType type;

  // Code expression that tests if the condition should be evaluated e.g.
  // "node->cond.feat == 2".
  // If "cond" is not set, create automatically a condition of the type
  // "condition_types[node->cond.feat] == condition_type".
  std::string used_code;

  // Code of the condition. Should set a "eval" boolean value, e.g.
  // "eval = eval = raw_numerical[node->cond.feat] >= node->cond.thr"
  std::string eval_code;
};

// Function signature to generate the tree inference code using the if-else
// algorithm.
typedef std::function<absl::StatusOr<std::string>(
    const model::decision_tree::proto::Node& node, int depth, int tree_idx,
    absl::string_view prefix)>
    IfElseSetNodeFn;

// Specification of the type and shape of a leaf value.
struct LeafValueSpec {
  proto::DType::Enum dtype = proto::DType::UNDEFINED;
  int dims = 0;
};

// Generic leaf value.
typedef std::variant<std::vector<int32_t>, std::vector<float>,
                     std::vector<bool>>
    LeafValue;

// Function signature that returns leaf values.
typedef std::function<LeafValue(const model::decision_tree::proto::Node& leaf)>
    LeafValueFn;

// Constants and functions specific to certain decision forest models.
struct SpecializedConversion {
  std::string accumulator_type;
  std::string accumulator_initial_value;
  std::string return_prediction;
  LeafValueSpec leaf_value_spec;
  IfElseSetNodeFn set_node_ifelse_fn;
  LeafValueFn leaf_value_fn;
  std::string routing_node;

  // Validate the object.
  absl::Status Validate() const;
};

// Reserved feature index used for oblique conditions.
int ObliqueFeatureIndex(const InternalOptions& internal_options);

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

// Computes the statistics of the model.
absl::StatusOr<ModelStatistics> ComputeStatistics(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface);

struct FeatureDef {
  std::string type;  // Type to encode a feature using typedef / enum class.
  std::string underlying_type;  // Type to encode a feature e.g. "float".
  absl::optional<std::string> default_value = {};  // Optional default value.
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
                         const ValueBank& routing_bank, std::string* content);

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

}  // namespace internal
}  // namespace yggdrasil_decision_forests::serving::embed

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_EMBED_H_
