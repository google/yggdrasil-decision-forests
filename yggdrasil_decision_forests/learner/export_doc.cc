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

#include "yggdrasil_decision_forests/learner/export_doc.h"

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {

namespace {

// Creates part of the markdown table showing the available generic hyper
// parameters.
//
// Creates the last two columns ("default value" and "range") for a numerical
// hyper-parameter (i.e. integer or real).
template <typename T>
std::string NumericalDefaultAndRangeForHyperParameterValueToMarkdownTable(
    const T& value) {
  std::string result;
  if (value.has_default_value()) {
    absl::SubstituteAndAppend(&result, "**Default:** $0",
                              value.default_value());
  }
  if (value.has_minimum() || value.has_maximum()) {
    absl::StrAppend(&result, " **Possible values:**");
  }

  if (value.has_minimum()) {
    absl::SubstituteAndAppend(&result, " min:$0", value.minimum());
  }
  if (value.has_maximum()) {
    absl::SubstituteAndAppend(&result, " max:$0", value.maximum());
  }
  return result;
}

// Sanitize markdown text to be injected into a markdown table.
std::string MarkDownInTable(std::string content) {
  content = absl::StrReplaceAll(content, {{"\n", "<br>"}});
  return content;
}

}  // namespace

utils::StatusOr<std::string> ExportSeveralLearnersToMarkdown(
    std::vector<std::string> learners,
    const DocumentationUrlFunctor& gen_doc_url,
    const std::vector<std::string>& ordering) {
  std::string content;

  LOG(INFO) << "Export learners:";
  for (const auto& x : learners) {
    LOG(INFO) << "\t" << x;
  }

  // Index of "v" in "ordering". Returns "ordering.size()" is "v" is not in
  // "ordering".
  auto get_index_in_ordering = [&](const std::string& v) {
    return std::distance(ordering.begin(),
                         std::find(ordering.begin(), ordering.end(), v));
  };

  // Sort the learners.
  std::sort(learners.begin(), learners.end(),
            [&](const std::string& a, const std::string& b) -> bool {
              const int index_a = get_index_in_ordering(a);
              const int index_b = get_index_in_ordering(b);
              if (index_a != index_b) {
                return index_a < index_b;
              } else if (index_a == ordering.size()) {
                return false;
              } else if (index_b == ordering.size()) {
                return true;
              } else {
                return a < b;
              }
            });

  for (const auto& learner_name : learners) {
    std::unique_ptr<AbstractLearner> learner;
    proto::TrainingConfig train_config;
    train_config.set_learner(learner_name);
    train_config.set_label("my_label");
    RETURN_IF_ERROR(GetLearner(train_config, &learner));
    ASSIGN_OR_RETURN(const auto specifications,
                     learner->GetGenericHyperParameterSpecification());
    ASSIGN_OR_RETURN(auto sub_content, ExportHParamSpecToMarkdown(
                                           learner->training_config().learner(),
                                           specifications, gen_doc_url));
    absl::StrAppendFormat(&content, "## %s\n\n%s\n\n", learner_name,
                          sub_content);
  }

  return content;
}

utils::StatusOr<std::string> ExportHParamSpecToMarkdown(
    absl::string_view learner_key,
    const proto::GenericHyperParameterSpecification& hparams,
    const DocumentationUrlFunctor& gen_doc_url) {
  // Check if the set of generic hyper-parameters is empty.
  // Note: "kHParamMaximumTrainingDurationSeconds" is added to most learners
  // automatically.

  bool no_generic_hparams = true;
  for (const auto& field : hparams.fields()) {
    if (field.first == kHParamMaximumTrainingDurationSeconds ||
        field.first == kHParamMaximumModelSizeInMemoryInBytes ||
        field.first == kHParamRandomSeed) {
      continue;
    }
    no_generic_hparams = false;
    break;
  }

  if (no_generic_hparams) {
    // The name of the proto filename, source filename, c++ name_space,
    // learner_key, and extension are all "learner_key".
    // Caveat: The learner_key uses upper cases.
    const std::string base_proto = absl::Substitute(
        "learner/$0.proto", absl::AsciiStrToLower(learner_key));
    return absl::Substitute(
        "No generic hyper-parameters. Use the <a href=\"$0\">$1</a> "
        "TrainingConfig proto instead.",
        gen_doc_url(base_proto, {}), learner_key);
  }

  std::string result;
  absl::StrAppend(&result, "<font size=\"2\">\n\n");

  // Introduction to the learner.
  if (hparams.documentation().has_description()) {
    absl::StrAppend(&result, hparams.documentation().description());
    absl::StrAppend(&result, "\n\n");
  }

  // List of all the protos.
  std::set<std::string> protos;
  for (const auto& field : hparams.fields()) {
    if (field.second.documentation().has_proto_path()) {
      protos.insert(field.second.documentation().proto_path());
    }
  }
  if (!protos.empty()) {
    absl::StrAppend(&result, "### Training configuration\n\n");
    for (const auto& proto : protos) {
      absl::SubstituteAndAppend(&result, "- <a href=\"$0\">$1</a>\n",
                                gen_doc_url(proto, {}), proto);
    }
    absl::StrAppend(&result, "\n");
  }

  absl::StrAppend(&result,
                  "### Generic Hyper-parameters (compatible with TensorFlow "
                  "Decision Forests)\n\n");

  // Sort the field alphabetically.
  std::vector<
      std::pair<std::string, proto::GenericHyperParameterSpecification::Value>>
      fields;
  for (const auto& hparam : hparams.fields()) {
    fields.emplace_back(hparam);
  }
  std::sort(fields.begin(), fields.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  // For each hyper-parameter.
  for (const auto& hparam : fields) {
    // Hyper parameter name.
    std::string url;
    if (hparam.second.documentation().has_proto_path()) {
      const auto field_name = hparam.second.documentation().has_proto_field()
                                  ? hparam.second.documentation().proto_field()
                                  : hparam.first;
      url = gen_doc_url(hparam.second.documentation().proto_path(), field_name);
    }

    if (url.empty()) {
      absl::SubstituteAndAppend(&result, "#### $0\n\n", hparam.first);
    } else {
      absl::SubstituteAndAppend(&result, "#### [$0]($1)\n\n", hparam.first,
                                url);
    }

    absl::StrAppend(&result, " - ");

    switch (hparam.second.Type_case()) {
      case proto::GenericHyperParameterSpecification::Value::kReal:
        absl::SubstituteAndAppend(&result, "**Type:** Real ");
        absl::StrAppend(
            &result,
            NumericalDefaultAndRangeForHyperParameterValueToMarkdownTable(
                hparam.second.real()));
        break;
      case proto::GenericHyperParameterSpecification::Value::kInteger:
        absl::SubstituteAndAppend(&result, "**Type:** Integer ");
        absl::StrAppend(
            &result,
            NumericalDefaultAndRangeForHyperParameterValueToMarkdownTable(
                hparam.second.integer()));
        break;
      case proto::GenericHyperParameterSpecification::Value::kCategorical: {
        const auto value = hparam.second.categorical();
        absl::SubstituteAndAppend(&result, "**Type:** Categorical");
        if (value.has_default_value()) {
          absl::SubstituteAndAppend(&result, " **Default:** $0",
                                    value.default_value());
        }
        absl::SubstituteAndAppend(&result, " **Possible values:** $0",
                                  absl::StrJoin(value.possible_values(), ", "));
      } break;
      case proto::GenericHyperParameterSpecification::Value::kCategoricalList: {
        const auto value = hparam.second.categorical();
        absl::SubstituteAndAppend(&result, "**Type:** Categorical list");
      } break;
      default:
        LOG(FATAL) << "Not implemented";
    }

    absl::SubstituteAndAppend(
        &result, "\n\n - $0\n\n",
        MarkDownInTable(hparam.second.documentation().description()));
  }
  absl::StrAppend(&result, "</font>\n");
  return result;
}

}  // namespace model
}  // namespace yggdrasil_decision_forests
