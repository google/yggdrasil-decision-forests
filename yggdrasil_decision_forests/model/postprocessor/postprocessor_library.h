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

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_POSTPROCESSOR_POSTPROCESSOR_LIBRARY_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_POSTPROCESSOR_POSTPROCESSOR_LIBRARY_H_

#include <memory>

#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/postprocessor/abstract_postprocessor.h"
#include "yggdrasil_decision_forests/model/postprocessor/abstract_postprocessor.pb.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace postprocessor {

absl::StatusOr<std::unique_ptr<AbstractPostprocessor>> CreatePostprocessor(
    const proto::AbstractPostprocessor& proto);

}  // namespace postprocessor
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_POSTPROCESSOR_POSTPROCESSOR_LIBRARY_H_
