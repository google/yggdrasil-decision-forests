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

#include "yggdrasil_decision_forests/serving/embed/c/c_embed.h"

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/serving/embed/c/c_emitter.h"
#include "yggdrasil_decision_forests/serving/embed/c/c_ir.h"
#include "yggdrasil_decision_forests/serving/embed/c/c_target_lowering.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"
#include "yggdrasil_decision_forests/serving/embed/ir/builder.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelC(
    const model::AbstractModel& model, const proto::Options& options) {
  if (options.algorithm() != proto::Algorithm::ROUTING) {
    return absl::InvalidArgumentError(
        "Export to C is only implemented for the ROUTING algorithm");
  }
  RETURN_IF_ERROR(CheckModelName(options.name(), proto::Options::kC));
  for (const auto& column_idx : model.input_features()) {
    RETURN_IF_ERROR(
        CheckFeatureName(model.data_spec().columns(column_idx).name()));
  }

  ASSIGN_OR_RETURN(internal::ModelIR ir,
                   internal::ModelIRBuilder::Build(model, options));

  ASSIGN_OR_RETURN(internal::CIR cpp_ir,
                   internal::CTargetLowering::Lower(ir, options));

  ASSIGN_OR_RETURN(const auto& generated_code,
                   internal::CEmitter::Emit(cpp_ir, options));

  return generated_code;
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
