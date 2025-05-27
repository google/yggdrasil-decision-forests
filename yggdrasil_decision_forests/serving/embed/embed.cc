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

#include "yggdrasil_decision_forests/serving/embed/embed.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed {

namespace {

// Converts any string into a snake case symbol.
std::string StringToSnakeCaseSymbol(const std::string_view input,
                                    const bool to_upper,
                                    const char prefix_char_if_digit) {
  if (input.empty()) {
    return "";
  }

  std::string result;
  result.reserve(input.size());
  bool last_char_was_separator = true;
  bool first_char = true;

  for (const char ch : input) {
    if (std::isalnum(ch)) {
      if (std::isdigit(ch) && first_char) {
        // Add a prefix if the first character is a number.
        result.push_back(prefix_char_if_digit);
      }
      // Change the case of letters.
      if (to_upper) {
        result.push_back(std::toupper(ch));
      } else {
        result.push_back(std::tolower(ch));
      }
      last_char_was_separator = false;
    } else if (ch == ' ' || ch == '-' || ch == '_') {
      // Characters that are replaced with "_".
      if (!result.empty() && !last_char_was_separator) {
        result.push_back('_');
        last_char_was_separator = true;
      }
    }
    // Other characters are skipped.
    first_char = false;
  }

  // Remove the last character if it is a separator.
  if (!result.empty() && result.back() == '_') {
    result.pop_back();
  }
  return result;
}

}  // namespace

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelCC(
    const model::AbstractModel& model, const proto::Options& options) {
  RETURN_IF_ERROR(internal::CheckModelName(options.name()));

  absl::node_hash_map<Filename, Content> result;

  result[absl::StrCat(options.name(), ".h")] =
      absl::Substitute(R"(#ifndef MODEL_$0_H_
#define MODEL_$0_H_
namespace $0 {
inline void f() {}
}  // namespace $0
#endif
)",
                       options.name());
  return result;
}

namespace internal {

absl::StatusOr<ModelStatistics> ComputeStatistics(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface) {
  ModelStatistics stats{
      .num_trees = df_interface.num_trees(),
      .num_features = static_cast<int>(model.input_features().size()),
  };

  // Model specific statistics.
  const auto model_gbt = dynamic_cast<
      const model::gradient_boosted_trees::GradientBoostedTreesModel*>(&model);
  const auto model_rf =
      dynamic_cast<const model::random_forest::RandomForestModel*>(&model);
  if (model_gbt) {
    stats.multi_dim_tree = false;
  } else if (model_rf) {
    stats.multi_dim_tree = true;
  } else {
    return absl::InvalidArgumentError("The model type is not supported.");
  }

  if (model.task() == model::proto::Task::CLASSIFICATION &&
      model.LabelColumnSpec().categorical().number_of_unique_values() > 3) {
    stats.internal_output_dim =
        model.LabelColumnSpec().categorical().number_of_unique_values() - 1;
  } else {
    stats.internal_output_dim = 1;
  }

  // Scan the trees
  for (const auto& tree : df_interface.decision_trees()) {
    int64_t num_leaves_in_tree = 0;
    tree->IterateOnNodes([&](const model::decision_tree::NodeWithChildren& node,
                             int depth) {
      stats.max_depth = std::max(stats.max_depth, static_cast<int64_t>(depth));
      if (node.IsLeaf()) {
        num_leaves_in_tree++;
        stats.num_leaves++;
      }
    });
    stats.max_num_leaves_per_tree =
        std::max(stats.max_num_leaves_per_tree, num_leaves_in_tree);
  }
  return stats;
}

absl::Status CheckModelName(absl::string_view value) {
  for (const char c : value) {
    if (!std::islower(c) && !std::isdigit(c) && c != '_') {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid model name: ", value,
                       ". The model name can only contain lowercase letters, "
                       "numbers, and _."));
    }
  }
  return absl::OkStatus();
}

std::string StringToConstantSymbol(const absl::string_view input) {
  return StringToSnakeCaseSymbol(input, true, 'V');
}

std::string StringToVariableSymbol(const absl::string_view input) {
  return StringToSnakeCaseSymbol(input, false, 'v');
}

std::string StringToStructSymbol(const absl::string_view input) {
  if (input.empty()) {
    return "";
  }

  std::string result;
  result.reserve(input.size());
  bool capitalize_next_char = true;
  bool current_word_started_with_letter = false;
  bool first_char = true;

  for (const char ch : input) {
    if (std::isalnum(ch)) {
      if (std::isdigit(ch) && first_char) {
        // Add a prefix if the first character is a number.
        result.push_back('V');
      }
      // Change the case of letters.
      if (capitalize_next_char) {
        result.push_back(std::toupper(ch));
        current_word_started_with_letter = std::isalpha(ch);
        capitalize_next_char = false;
      } else {
        if (current_word_started_with_letter && std::isalpha(ch)) {
          result.push_back(std::tolower(ch));
        } else {
          result.push_back(ch);
        }
      }
    } else {
      capitalize_next_char = true;
    }
    // Other characters are skipped.
    first_char = false;
  }

  // Remove the last character if it is a separator.
  if (!result.empty() && result.back() == '_') {
    result.pop_back();
  }
  return result;
}

}  // namespace internal
}  // namespace yggdrasil_decision_forests::serving::embed
