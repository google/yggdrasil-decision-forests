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

#include "yggdrasil_decision_forests/utils/protobuf.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/google/protobuf/message.h"
#include "src/google/protobuf/text_format.h"

namespace yggdrasil_decision_forests::utils {

absl::StatusOr<std::string> SerializeTextProto(const google::protobuf::Message& message,
                                               bool single_line_mode) {
  std::string serialized_message;
  google::protobuf::TextFormat::Printer printer;
  if (single_line_mode) {
    printer.SetSingleLineMode(true);
  }
  if (!printer.PrintToString(message, &serialized_message)) {
    return absl::InvalidArgumentError("Cannot serialize proto message.");
  }
  if (single_line_mode && !serialized_message.empty() &&
      serialized_message.back() == ' ') {
    serialized_message.pop_back();
  }
  return serialized_message;
}

}  // namespace yggdrasil_decision_forests::utils
