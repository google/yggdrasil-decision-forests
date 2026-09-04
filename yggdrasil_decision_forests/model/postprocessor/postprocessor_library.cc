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

#include "yggdrasil_decision_forests/model/postprocessor/postprocessor_library.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/postprocessor/abstract_postprocessor.h"
#include "yggdrasil_decision_forests/model/postprocessor/postprocessor.pb.h"
#include "yggdrasil_decision_forests/model/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace postprocessor {

absl::StatusOr<std::unique_ptr<AbstractPostprocessor>> CreatePostprocessor(
    const proto::Postprocessor& proto) {
  switch (proto.postprocessor_case()) {
    case proto::Postprocessor::kSmoothedPavCalibrator:
      return std::make_unique<SmoothedPavCalibrator>(
          proto.smoothed_pav_calibrator());
    default:
      return absl::InvalidArgumentError("Unknown postprocessor type.");
  }
}

}  // namespace postprocessor
}  // namespace model
}  // namespace yggdrasil_decision_forests
