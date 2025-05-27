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
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"

namespace yggdrasil_decision_forests::serving::embed {

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelCC(
    const model::AbstractModel& model, const proto::Options& options) {
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

}  // namespace yggdrasil_decision_forests::serving::embed
