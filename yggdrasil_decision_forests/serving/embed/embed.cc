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

#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/serving/embed/cc_embed.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/java_embed.h"

namespace yggdrasil_decision_forests::serving::embed {

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModel(
    const model::AbstractModel& model, const proto::Options& options) {
  switch (options.language_case()) {
    case proto::Options::kCc:
      return internal::EmbedModelCC(model, options);
    case proto::Options::kJava:
      return internal::EmbedModelJava(model, options);
    case proto::Options::LANGUAGE_NOT_SET:
      return absl::InvalidArgumentError(
          "No language for export set, options are CC, Java");
  }
  LOG(ERROR) << "Unexpected value for Language: "
             << static_cast<int>(options.language_case());
  return absl::InternalError("Unexpected value for Language");
}

}  // namespace yggdrasil_decision_forests::serving::embed
