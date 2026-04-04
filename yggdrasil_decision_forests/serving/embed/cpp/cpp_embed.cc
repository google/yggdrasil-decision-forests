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

#include "yggdrasil_decision_forests/serving/embed/cpp/cpp_embed.h"

#include <string>

#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/cpp_emitter.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/cpp_ir.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/cpp_target_lowering.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/ir/builder.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelCpp(
    const model::AbstractModel& model, const proto::Options& options) {
  RETURN_IF_ERROR(CheckModelName(options.name(), proto::Options::kCpp));
  for (const auto& column_idx : model.input_features()) {
    RETURN_IF_ERROR(
        CheckFeatureName(model.data_spec().columns(column_idx).name()));
  }

  ASSIGN_OR_RETURN(internal::ModelIR ir,
                   internal::ModelIRBuilder::Build(model, options));

  ASSIGN_OR_RETURN(internal::CppIR cpp_ir,
                   internal::CppTargetLowering::Lower(ir, options));

  ASSIGN_OR_RETURN(std::string generated_code,
                   internal::CppEmitter::Emit(cpp_ir, options));

  // Generate the code.
  absl::node_hash_map<Filename, Content> result;
  const std::string filename = absl::StrCat(options.name(), ".h");
  result[filename] = generated_code;

  return result;
}
}  // namespace yggdrasil_decision_forests::serving::embed::internal
