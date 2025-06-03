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
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
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
  // Maximum number of leaves among all the trees.
  int64_t max_num_leaves_per_tree = 0;
  // Maximum depth. max_depth=0 indicates that the tree contains a single node.
  int64_t max_depth = 0;
  // Number of input features.
  int num_features = 0;
  // Number of outputs of the forest i.e., before any possibly reduction by the
  // compiled model. For instance, a 3 classes classification model will have
  // "internal_output_dim=3".
  int internal_output_dim = 0;
  // Are individual trees returning multidimensional outputs.
  bool multi_dim_tree = false;

  // If the individual trees can output a negative value.
  bool leaf_output_is_signed = true;

  // Maximum of the absolute value of the node outputs over all the trees and
  // nodes.
  double max_abs_output = 0;

  // Sum over all the trees, of the maximum absolute values over all the
  // nodes.
  double sum_max_abs_output = 0;

  // Which conditions are used by the model.
  std::array<bool, model::decision_tree::kNumConditionTypes + 1> has_conditions{
      false};
};

// Specific options for the generation of the model.
// The internal options contains all the precise internal decision aspect of the
// model compilation e.g. how many bits to use to encode numerical features. The
// internal options are computed using the user provided options (simply called
// "options" in the code) and the model.
struct InternalOptions {
  // Number of bits to encode a fixed-size feature.
  // Note: Currently, all the fixed-size features are encoded with the same
  // precision (e.g. all the numerical and categorical values are encoded with
  // the same number of bits). Can be 1, 2, or 4.
  int feature_value_bytes = 0;

  // If the numerical features are encoded as float. In this case
  // feature_value_bits=4 (currently). If false, numerical features are encoded
  // as ints, and "feature_value_bytes" specify the precision.
  bool numerical_feature_is_float = false;

  // The type returned by the prediction function.
  std::string output_type;

  // Type of the accumulator to accumulate the leaf values.
  std::string accumulator_type;

  // Type of the leaf values.
  std::string leaf_value_type;

  // If true, the model requires the <array> include.
  bool include_array = false;
  // If true, the model requires the <algorithm> include.
  bool include_algorithm = false;
  // If true, the model requires the <cmath> include.
  bool include_cmath = false;

  // Coefficient applied on the numerical leaf values. Only use when
  // "integerize_output=true" and if the tree leaves contain numerical values.
  std::optional<double> coefficient;

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

// Computes the internal options of the model.
absl::StatusOr<InternalOptions> ComputeInternalOptions(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface,
    const ModelStatistics& stats, const proto::Options& options);

// Populates the feature parts of the internal option.
absl::Status ComputeInternalOptionsFeature(const model::AbstractModel& model,
                                           const proto::Options& options,
                                           InternalOptions* out);

// Populates the output parts of the internal option.
absl::Status ComputeInternalOptionsOutput(const model::AbstractModel& model,
                                          const ModelStatistics& stats,
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
  std::string type;  // Type to encode a feature e.g. "float".
  absl::optional<std::string> default_value = {};  // Optional default value.
};

// Generates the definition of a feature in an instance struct.
absl::StatusOr<FeatureDef> GenFeatureDef(
    const dataset::proto::Column& col,
    const internal::InternalOptions& internal_options);

// Generation of the prediction code for a GBT model.
absl::Status GenPredictionGBT(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& model,
    const ModelStatistics& stats, const InternalOptions& internal_options,
    const proto::Options& options, std::string* content);

// Generation of the prediction code for a RF model.
absl::Status GenPredictionRF(
    const model::random_forest::RandomForestModel& model,
    const ModelStatistics& stats, const InternalOptions& internal_options,
    const proto::Options& options, std::string* content);

// The scalar type of an accumulator.
struct AccumulatorDef {
  std::string type;
  std::string base_type;
  bool use_array = false;
};
AccumulatorDef GenAccumulatorDef(const proto::Options& options,
                                 const ModelStatistics& stats);

// Generates the tree inference code using the if-else algorithm.
typedef std::function<absl::StatusOr<std::string>(
    const model::decision_tree::proto::Node& noden, int depth, int tree_idx,
    absl::string_view prefix)>
    IfElseSetNodeFn;

absl::Status GenerateTreeInferenceIfElse(
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const proto::Options& options, const InternalOptions& internal_options,
    const IfElseSetNodeFn& set_node_fn, std::string* content);

}  // namespace internal
}  // namespace yggdrasil_decision_forests::serving::embed

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_EMBED_H_
