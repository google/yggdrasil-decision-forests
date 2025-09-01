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

#include "yggdrasil_decision_forests/serving/embed/java_embed.h"

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelJava(
    const model::AbstractModel& model, const proto::Options& options) {
  return absl::UnimplementedError("Java export not yet implemented.");
}
}  // namespace yggdrasil_decision_forests::serving::embed::internal
