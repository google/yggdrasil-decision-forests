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

// A binary that writes the result of an embedded model to disk.
// This binary is used by the "cc_ydf_standalone_model " build rule.

#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/embed/embed.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

ABSL_FLAG(
    std::string, name, "",
    "Name of the model. If set in the options proto, this value is ignored.");
ABSL_FLAG(std::string, input, "", "Input YDF model directory");
ABSL_FLAG(std::string, output, "",
          "Output directory where to write the generated files.");
ABSL_FLAG(std::string, options, "", "Options for the embedded model");
ABSL_FLAG(bool, remove_output_filename, false,
          "If set, 'output' is a file path. The filename should be removed to "
          "get the real output directory.");
ABSL_FLAG(
    std::string, language, "CC",
    "Target language. If set in the options proto, this value is ignored.");

namespace yggdrasil_decision_forests::serving::embed {

absl::Status WriteEmbeddedModel() {
  const auto name = absl::GetFlag(FLAGS_name);
  const auto input = absl::GetFlag(FLAGS_input);
  std::string output = absl::GetFlag(FLAGS_output);
  const auto remove_output_filename =
      absl::GetFlag(FLAGS_remove_output_filename);
  const auto options_text = absl::GetFlag(FLAGS_options);
  const auto language_str = absl::GetFlag(FLAGS_language);

  if (remove_output_filename) {
    output = file::GetDirname(output);
  }

  LOG(INFO) << "Compiling model";
  proto::Options options;
  if (!options_text.empty()) {
    ASSIGN_OR_RETURN(options,
                     utils::ParseTextProto<proto::Options>(options_text));
  }
  if (!options.has_name()) {
    options.set_name(name);
  }

  if (options.language_case() == proto::Options::LANGUAGE_NOT_SET) {
    if (language_str == "CC") {
      options.mutable_cc();
    } else if (language_str == "Java") {
      options.mutable_java();
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unknown language ", language_str, ". Available options: CC, Java"));
    }
  }

  LOG(INFO) << "Loading model";
  ASSIGN_OR_RETURN(const std::unique_ptr<model::AbstractModel> model,
                   model::LoadModel(input));

  ASSIGN_OR_RETURN(const auto embedded_model, EmbedModel(*model, options));

  LOG(INFO) << "Write embedded model";
  for (const auto& file_and_data : embedded_model) {
    RETURN_IF_ERROR(file::SetContent(
        file::JoinPath(output, file_and_data.first), file_and_data.second));
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::serving::embed

int main(int argc, char** argv) {
  InitLogging("", &argc, &argv, true);
  QCHECK_OK(yggdrasil_decision_forests::serving::embed::WriteEmbeddedModel());
  return 0;
}
