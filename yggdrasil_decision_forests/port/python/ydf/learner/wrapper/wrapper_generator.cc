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

#include "ydf/learner/wrapper/wrapper_generator.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/model/hyperparameter.pb.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {

// Gets the number of prefix spaces.
int NumLeadingSpaces(const absl::string_view text) {
  auto char_it = text.begin();
  while (char_it != text.end() && *char_it == ' ') {
    char_it++;
  }
  return std::distance(text.begin(), char_it);
}

// Converts a learner name into a python class name.
// e.g. RANDOM_FOREST -> RandomForestLearner
std::string LearnerKeyToClassName(const absl::string_view key) {
  std::string value(key);
  for (auto it = value.begin(); it != value.end(); ++it) {
    if (it == value.begin() || !absl::ascii_isalpha(*(it - 1))) {
      *it = absl::ascii_toupper(*it);
    } else {
      *it = absl::ascii_tolower(*it);
    }
  }
  return absl::StrCat(absl::StrReplaceAll(value, {{"_", ""}}), "Learner");
}

// Converts a learner name into a nice name.
// e.g. "RANDOM_FOREST" -> "Random Forest"
std::string LearnerKeyToNiceLearnerName(absl::string_view key) {
  std::string value(key);
  for (auto it = value.begin(); it != value.end(); ++it) {
    if (it == value.begin() || !absl::ascii_isalpha(*(it - 1))) {
      *it = absl::ascii_toupper(*it);
    } else {
      *it = absl::ascii_tolower(*it);
    }
  }
  return absl::StrReplaceAll(value, {{"_", " "}});
}

// Converts a floating point value to its python representation.
std::string PythonFloat(const float value) {
  std::string str_value;
  absl::StrAppendFormat(&str_value, "%g", value);
  // Check if the number is finite and written in decimal notation.
  if (std::isfinite(value) && !absl::StrContains(str_value, "+")) {
    // Make sure the value is a python floating point number.
    if (!absl::StrContains(str_value, ".")) {
      absl::StrAppend(&str_value, ".0");
    }
  }
  return str_value;
}

// Generates the Python object for the pre-defined hyper-parameters and the name
// of the first template for the documentation.
absl::StatusOr<std::pair<std::string, std::string>>
BuildHyperparameterTemplates(const model::AbstractLearner* learner) {
  // Python dictionary of template hyper-parameters.
  std::string predefined_hp_dict = "{";
  std::string first_template_name = "";

  const auto hyperparameter_templates = learner->PredefinedHyperParameters();
  for (const auto& hp_template : hyperparameter_templates) {
    if (first_template_name.empty()) {
      first_template_name =
          absl::Substitute("$0v$1", hp_template.name(), hp_template.version());
    }
    absl::SubstituteAndAppend(
        &predefined_hp_dict,
        "\"$0v$1\": "
        "hyperparameters.HyperparameterTemplate(name=\"$0\", "
        "version=$1, description=\"$2\", parameters={",
        hp_template.name(), hp_template.version(),
        absl::StrReplaceAll(hp_template.description(), {{"\"", "\\\""}}));
    // Iterate over the individual parameters.
    bool first_field = true;
    for (const auto& field : hp_template.parameters().fields()) {
      if (first_field) {
        first_field = false;
      } else {
        absl::StrAppend(&predefined_hp_dict, ", ");
      }
      absl::StrAppend(&predefined_hp_dict, "\"", field.name(), "\" :");
      switch (field.value().Type_case()) {
        case model::proto::GenericHyperParameters_Value::TYPE_NOT_SET:
          return absl::InternalError("Non configured value");
          break;
        case model::proto::GenericHyperParameters_Value::kCategorical: {
          std::string value = field.value().categorical();
          if (value == "true" || value == "false") {
            value = (value == "true") ? "True" : "False";
          } else {
            value = absl::StrCat("\"", value, "\"");
          }
          absl::StrAppend(&predefined_hp_dict, value);
        } break;
        case model::proto::GenericHyperParameters_Value::kInteger:
          absl::StrAppend(&predefined_hp_dict, field.value().integer());
          break;
        case model::proto::GenericHyperParameters_Value::kReal:
          absl::StrAppend(&predefined_hp_dict,
                          PythonFloat(field.value().real()));
          break;
        case model::proto::GenericHyperParameters_Value::kCategoricalList:
          absl::StrAppend(
              &predefined_hp_dict, "[",
              absl::StrJoin(field.value().categorical_list().values(), ","),
              "]");
          break;
      }
    }
    absl::SubstituteAndAppend(&predefined_hp_dict, "}), ");
  }
  absl::StrAppend(&predefined_hp_dict, "}");
  return std::make_pair(predefined_hp_dict, first_template_name);
}

// Formats some documentation.
//
// Args:
//   raw: Raw documentation.
//   leading_spaces_first_line: Left margin on the first line.
//   leading_spaces_next_lines: Left margin on the next lines.
//   max_char_per_lines: Maximum line length for word wrapping.
//
std::string FormatDocumentation(const absl::string_view raw,
                                const int leading_spaces_first_line,
                                const int leading_spaces_next_lines,
                                const int max_char_per_lines) {
  // Number of spaces to create an "offset".
  const int offset = 2;
  // Sanitize documentation.
  std::string raw_sanitized = absl::StrReplaceAll(raw, {{"\\", "\\\\"}});

  // Extract the lines of text.
  const std::vector<std::string> lines = absl::StrSplit(raw_sanitized, '\n');
  std::string formatted;

  for (int line_idx = 0; line_idx < lines.size(); line_idx++) {
    const auto& line = lines[line_idx];

    // Leading spaces of the current line.
    const int user_leading_spaces = NumLeadingSpaces(line);

    // Detect a list.
    const bool bullet_list = line.size() >= 2 && line.substr(0, 2) == "- ";
    /*
        const bool definition_header =
            line_idx == 0;  // absl::StrContains(line, ":");
    */
    const auto leading_spaces =
        (line_idx == 0) ? leading_spaces_first_line : leading_spaces_next_lines;

    int written_length = leading_spaces + user_leading_spaces;
    absl::StrAppend(&formatted, std::string(written_length, ' '));

    const std::vector<std::string> tokens = absl::StrSplit(line, ' ');
    for (int token_idx = 0; token_idx < tokens.size(); token_idx++) {
      const auto& token = tokens[token_idx];
      if (written_length + token.size() + 1 > max_char_per_lines) {
        // Wrap the line.
        written_length = leading_spaces_next_lines + user_leading_spaces;
        if (bullet_list /*|| definition_header*/) {
          written_length += offset;
        }
        absl::StrAppend(&formatted, "\n");
        absl::StrAppend(&formatted, std::string(written_length, ' '));
      } else if (token_idx > 0) {
        absl::StrAppend(&formatted, " ");
      }
      absl::StrAppend(&formatted, token);
      written_length += token.size() + 1;
    }

    // Tailing line return.
    absl::StrAppend(&formatted, "\n");
  }
  return formatted;
}

absl::StatusOr<std::string> GenLearnerWrapper() {
  const auto prefix = "";
  const auto pydf_prefix = "ydf.";

  std::string imports = absl::Substitute(R"(
from $0yggdrasil_decision_forests.dataset import data_spec_pb2
from $0yggdrasil_decision_forests.learner import abstract_learner_pb2
from $1dataset import dataspec
from $1dataset import dataset
from $1learner import generic_learner
from $1learner import hyperparameters
from $1learner import tuner as tuner_lib
)",
                                         prefix, pydf_prefix);

  std::string wrapper =
      absl::Substitute(R"(r"""Wrappers around the YDF learners.

This file is generated automatically by running the following commands:
  bazel build //ydf/learner:specialized_learners\
  && bazel-bin/ydf/learner/specialized_learners_generator\
  > ydf/learner/specialized_learners_pre_generated.py

Please don't change this file directly. Instead, changes the source. The
documentation source is contained in the "GetGenericHyperParameterSpecification"
method of each learner e.g. GetGenericHyperParameterSpecification in
learner/gradient_boosted_trees/gradient_boosted_trees.cc contains the
documentation (and meta-data) used to generate this file.

In particular, these pre-generated wrappers included in the source code are 
included for reference only. The actual wrappers are re-generated during
compilation.
"""

from typing import Dict, Optional, Sequence
$0

)",
                       imports);

  for (const auto& learner_key : model::AllRegisteredLearners()) {
    const auto class_name = LearnerKeyToClassName(learner_key);

    // Get a learner instance.
    std::unique_ptr<model::AbstractLearner> learner;
    model::proto::TrainingConfig train_config;
    train_config.set_learner(learner_key);
    train_config.set_label("my_label");
    RETURN_IF_ERROR(GetLearner(train_config, &learner));
    ASSIGN_OR_RETURN(const auto specifications,
                     learner->GetGenericHyperParameterSpecification());

    // Python documentation.
    std::string fields_documentation;
    // Constructor arguments.
    std::string fields_constructor;
    // Use of constructor arguments the parameter dictionary.
    std::string fields_dict;

    // Sort the fields alphabetically.
    std::vector<std::string> field_names;
    field_names.reserve(specifications.fields_size());
    for (const auto& field : specifications.fields()) {
      field_names.push_back(field.first);
    }
    std::sort(field_names.begin(), field_names.end());

    for (const auto& field_name : field_names) {
      const auto& field_def = specifications.fields().find(field_name)->second;

      if (field_def.documentation().deprecated()) {
        // Deprecated fields are not exported.
        continue;
      }

      // Constructor argument.
      if (!fields_constructor.empty()) {
        absl::StrAppend(&fields_constructor, ",\n");
      }
      // Type of the attribute.
      std::string attr_py_type;
      // Default value of the attribute.
      std::string attr_py_default_value;

      if (utils::HyperParameterIsBoolean(field_def)) {
        // Boolean values are stored as categorical.
        attr_py_type = "bool";
        attr_py_default_value =
            (field_def.categorical().default_value() == "true") ? "True"
                                                                : "False";
      } else {
        switch (field_def.Type_case()) {
          case model::proto::GenericHyperParameterSpecification::Value::
              kCategorical: {
            attr_py_type = "str";
            absl::SubstituteAndAppend(&attr_py_default_value, "\"$0\"",
                                      field_def.categorical().default_value());
          } break;
          case model::proto::GenericHyperParameterSpecification::Value::
              kInteger:
            attr_py_type = "int";
            absl::StrAppend(&attr_py_default_value,
                            field_def.integer().default_value());
            break;
          case model::proto::GenericHyperParameterSpecification::Value::kReal:
            attr_py_type = "float";
            absl::StrAppend(&attr_py_default_value,
                            PythonFloat(field_def.real().default_value()));
            break;
          case model::proto::GenericHyperParameterSpecification::Value::
              kCategoricalList:
            attr_py_type = "List[str]";
            attr_py_default_value = "None";
            break;
          case model::proto::GenericHyperParameterSpecification::Value::
              TYPE_NOT_SET:
            return absl::InvalidArgumentError(
                absl::Substitute("Missing type for field $0", field_name));
        }
      }

      // If the parameter is conditional on a parent parameter values, and the
      // default value of the parent parameter does not satisfy the condition,
      // the default value is set to None.
      if (field_def.has_conditional()) {
        const auto& conditional = field_def.conditional();
        const auto& parent_field =
            specifications.fields().find(conditional.control_field());
        if (parent_field == specifications.fields().end()) {
          return absl::InvalidArgumentError(
              absl::Substitute("Unknown conditional field $0 for field $1",
                               conditional.control_field(), field_name));
        }
        ASSIGN_OR_RETURN(
            const auto condition,
            utils::SatisfyDefaultCondition(parent_field->second, conditional));
        if (!condition) {
          attr_py_default_value = "None";
        }
      }

      // Constructor argument.
      absl::SubstituteAndAppend(&fields_constructor,
                                "      $0: Optional[$1] = $2", field_name,
                                attr_py_type, attr_py_default_value);

      // Assignation to parameter dictionary.
      absl::SubstituteAndAppend(
          &fields_dict, "                      \"$0\" : $0,\n", field_name);

      // Documentation
      if (field_def.documentation().description().empty()) {
        // Refer to the proto.
        absl::SubstituteAndAppend(&fields_documentation, "    $0: See $1\n",
                                  field_name,
                                  field_def.documentation().proto_path());
      } else {
        // Actual documentation.
        absl::StrAppend(
            &fields_documentation,
            FormatDocumentation(
                absl::StrCat(field_name, ": ",
                             field_def.documentation().description(),
                             " Default: ", attr_py_default_value, "."),
                /*leading_spaces_first_line=*/4,
                /*leading_spaces_next_lines=*/6));
      }
    }

    // Pre-configured hyper-parameters.
    std::string hp_template_dict;
    std::string first_template_name;
    ASSIGN_OR_RETURN(std::tie(hp_template_dict, first_template_name),
                     BuildHyperparameterTemplates(learner.get()));

    const auto free_text_documentation =
        FormatDocumentation(specifications.documentation().description(),
                            /*leading_spaces_first_line=*/2 - 2,
                            /*leading_spaces_next_lines=*/2);

    const auto nice_learner_name = LearnerKeyToNiceLearnerName(learner_key);

    // TODO: Add support for hyperparameter templates.
    absl::SubstituteAndAppend(&wrapper, R"(
class $0(generic_learner.GenericLearner):
  r"""$6 learning algorithm.

  $5
  Usage example:

  ```python
  import ydf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")

  model = ydf.$0().train(dataset)

  print(model.summary())
  ```

  Hyperparameters are configured to give reasonable results for typical
  datasets. Hyperparameters can also be modified manually (see descriptions)
  below or by applying the hyperparameter templates available with
  `$0.hyperparameter_templates()` (see this function's documentation for
  details).

  Attributes:
    label: Label of the dataset. The label column
      should not be identified as a feature in the `features` parameter.
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT).
    weights: Name of a feature that identifies the weight of each example. If
      weights are not specified, unit weights are assumed. The weight column
      should not be identified as a feature in the `features` parameter.
    ranking_group: Only for `task=Task.RANKING`. Name of a feature
      that identifies queries in a query/document ranking task. The ranking
      group should not be identified as a feature in the `features` parameter.
    uplift_treatment: Only for `task=Task.CATEGORICAL_UPLIFT` and `task=Task`.
      NUMERICAL_UPLIFT. Name of a numerical feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment. Currently, only 0/1 binary treatments are supported.
    features: If None, all columns are used as features. The semantic of the
      features is determined automatically. Otherwise, if
      include_all_columns=False (default) only the column listed in `features`
      are imported. If include_all_columns=True, all the columns are imported as
      features and only the semantic of the columns NOT in `columns` is 
      determined automatically. If specified,  defines the order of the features
      - any non-listed features are appended in-order after the specified
      features (if include_all_columns=True).
      The label, weights, uplift treatment and ranking_group columns should not
      be specified as features.
    include_all_columns: See `features`.
    max_vocab_count: Maximum size of the vocabulary of CATEGORICAL and
      CATEGORICAL_SET columns stored as strings. If more unique values exist,
      only the most frequent values are kept, and the remaining values are
      considered as out-of-vocabulary.
    min_vocab_frequency: Minimum number of occurrence of a value for CATEGORICAL
      and CATEGORICAL_SET columns. Value observed less than
      `min_vocab_frequency` are considered as out-of-vocabulary.
    discretize_numerical_columns: If true, discretize all the numerical columns
      before training. Discretized numerical columns are faster to train with,
      but they can have a negative impact on the model quality. Using
      `discretize_numerical_columns=True` is equivalent as setting the column
      semantic DISCRETIZED_NUMERICAL in the `column` argument. See the
      definition of DISCRETIZED_NUMERICAL for more details.
    num_discretized_numerical_bins: Number of bins used when disretizing
      numerical columns.
    data_spec: Dataspec to be used (advanced). If a data spec is given,
      `columns`, `include_all_columns`, `max_vocab_count`,
      `min_vocab_frequency`, `discretize_numerical_columns` and 
      `num_discretized_numerical_bins` will be ignored.
$2
    num_threads: Number of threads used to train the model. Different learning
      algorithms use multi-threading differently and with different degree of
      efficiency. If `None`, `num_threads` will be automatically set to the
      number of processors (up to a maximum of 32; or set to 6 if the number of
      processors is not available). Making `num_threads` significantly larger
      than the number of processors can slow-down the training speed. The
      default value logic might change in the future.
    resume_training: If true, the model training resumes from the checkpoint
      stored in the `working_dir` directory. If `working_dir` does not
      contain any model checkpoint, the training starts from the beginning.
      Resuming training is useful in the following situations: (1) The training
      was interrupted by the user (e.g. ctrl+c or "stop" button in a notebook)
      or rescheduled, or (2) the hyper-parameter of the learner was changed e.g.
      increasing the number of trees.
    working_dir: Path to a directory available for the learning algorithm to
      store intermediate computation results. Depending on the learning
      algorithm and parameters, the working_dir might be optional, required, or
      ignored. For instance, distributed training algorithm always need a
      "working_dir", and the gradient boosted tree and hyper-parameter tuners
      will export artefacts to the "working_dir" if provided.
    resume_training_snapshot_interval_seconds: Indicative number of seconds in 
      between snapshots when `resume_training=True`. Might be ignored by
      some learners.
    tuner: If set, automatically select the best hyperparameters using the
      provided tuner. When using distributed training, the tuning is
      distributed.
    workers: If set, enable distributed training. "workers" is the list of IP
      addresses of the workers. A worker is a process running
      `ydf.start_worker(port)`.
  """

  def __init__(self,
      label: str,
      task: generic_learner.Task = generic_learner.Task.CLASSIFICATION,
      weights: Optional[str] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      features: dataspec.ColumnDefs = None,
      include_all_columns: bool = False,
      max_vocab_count: int = 2000,
      min_vocab_frequency: int = 5,
      discretize_numerical_columns: bool = False,
      num_discretized_numerical_bins: int = 255,
      data_spec: Optional[data_spec_pb2.DataSpecification] = None,
$3,
      num_threads: Optional[int] = None,
      working_dir: Optional[str] = None,
      resume_training: bool = False,
      resume_training_snapshot_interval_seconds: int = 1800,
      tuner: Optional[tuner_lib.AbstractTuner] = None,
      workers: Optional[Sequence[str]] = None,
      ):

    hyper_parameters = {
$4
      }

    data_spec_args = dataspec.DataSpecInferenceArgs(
        columns=dataspec.normalize_column_defs(features),
        include_all_columns=include_all_columns,
        max_vocab_count=max_vocab_count,
        min_vocab_frequency=min_vocab_frequency,
        discretize_numerical_columns=discretize_numerical_columns,
        num_discretized_numerical_bins=num_discretized_numerical_bins
    )

    deployment_config = self._build_deployment_config(
        num_threads=num_threads,
        resume_training=resume_training,
        resume_training_snapshot_interval_seconds=resume_training_snapshot_interval_seconds,
        working_dir=working_dir,
        workers=workers,
    )

    super().__init__(learner_name="$1",
      task=task,
      label=label,
      weights=weights,
      ranking_group=ranking_group,
      uplift_treatment=uplift_treatment,
      data_spec_args=data_spec_args,
      data_spec=data_spec,
      hyper_parameters=hyper_parameters,
      deployment_config=deployment_config,
      tuner=tuner,
    )
)",
                              /*$0*/ class_name, /*$1*/ learner_key,
                              /*$2*/ fields_documentation,
                              /*$3*/ fields_constructor, /*$4*/ fields_dict,
                              /*$5*/ free_text_documentation,
                              /*$6*/ nice_learner_name);

    const auto bool_rep = [](const bool value) -> std::string {
      return value ? "True" : "False";
    };

    const auto capabilities = learner->Capabilities();
    absl::SubstituteAndAppend(
        &wrapper, R"(
  @classmethod
  def capabilities(cls) -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
      support_max_training_duration=$0,
      resume_training=$1,
      support_validation_dataset=$2,
      support_partial_cache_dataset_format=$3,
      support_max_model_size_in_memory=$4,
      support_monotonic_constraints=$5,
    )
)",
        /*$0*/ bool_rep(capabilities.support_max_training_duration()),
        /*$1*/ bool_rep(capabilities.resume_training()),
        /*$2*/ bool_rep(capabilities.support_validation_dataset()),
        /*$3*/ bool_rep(capabilities.support_partial_cache_dataset_format()),
        /*$4*/ bool_rep(capabilities.support_max_model_size_in_memory()),
        /*$5*/ bool_rep(capabilities.support_monotonic_constraints()));

    if (hp_template_dict == "{}") {
      absl::StrAppend(&wrapper, R"(
  @classmethod
  def hyperparameter_templates(cls) -> Dict[str, hyperparameters.HyperparameterTemplate]:
    r"""Hyperparameter templates for this Learner.
    
    This learner currently does not provide any hyperparameter templates, this
    method is provided for consistency with other learners.
    
    Returns:
      Empty dictionary.
    """
    return {}
)");
    } else {
      absl::SubstituteAndAppend(&wrapper, R"(
  @classmethod
  def hyperparameter_templates(cls) -> Dict[str, hyperparameters.HyperparameterTemplate]:
    r"""Hyperparameter templates for this Learner.
    
    Hyperparameter templates are sets of pre-defined hyperparameters for easy
    access to different variants of the learner. Each template is a mapping to a
    set of hyperparameters and can be applied directly on the learner.
    
    Usage example:
    ```python
    templates = ydf.$1.hyperparameter_templates()
    $2 = templates["$2"]
    # Print a description of the template
    print($2.description)
    # Apply the template's settings on the learner.
    learner = ydf.$1(label, **$2)
    ```
    
    Returns:
      Dictionary of the available templates
    """
    return $0
)",
                                /*$0*/ hp_template_dict, /*$1*/ class_name,
                                /*$2*/ first_template_name);
    }
  }

  return wrapper;
}

}  // namespace yggdrasil_decision_forests
