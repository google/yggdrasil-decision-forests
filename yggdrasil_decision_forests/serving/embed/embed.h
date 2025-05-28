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

#include <cstdint>
#include <string>

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
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
};

// Computes the internal options of the model.
absl::StatusOr<InternalOptions> ComputeInternalOptions(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface,
    const ModelStatistics& stats, const proto::Options& options);

// Computes the statistics of the model.
absl::StatusOr<ModelStatistics> ComputeStatistics(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface);

// Checks that a model name is valid. A model name can only contain letters,
// numbers, and _.
absl::Status CheckModelName(absl::string_view value);

// Converts any string into a c++ constant (with the "k") e.g. "HELLO_WOLRD_1".
std::string StringToConstantSymbol(absl::string_view input);

// Converts any string into a c++ variable name e.g. "hello_world_1".
std::string StringToVariableSymbol(absl::string_view input);

// Converts any string into a c++ struct name e.g. "HelloWorld1".
std::string StringToStructSymbol(absl::string_view input);

// Computes the number of bytes to encode the unsigned value. Can return 1, 2,
// or 4. For example, "MaxUnsignedValueToNumBytes" returns 2 for value=600
// (since using a single byte cannot encode a value greater than 255).
int MaxUnsignedValueToNumBytes(uint32_t value);

struct FeatureDef {
  std::string type;  // Type to encode a feature e.g. "float".
  absl::optional<std::string> default_value = {};  // Optional default value.
};

// Generates the definition of a feature in an instance struct.
absl::StatusOr<FeatureDef> GenFeatureDef(
    const dataset::proto::Column& col,
    const internal::InternalOptions& internal_options);

}  // namespace internal
}  // namespace yggdrasil_decision_forests::serving::embed

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_EMBED_H_
