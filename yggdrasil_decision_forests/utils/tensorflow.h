/*
 * Copyright 2021 Google LLC.
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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_TENSORFLOW_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_TENSORFLOW_H_

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/status.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Converts tensorflow::Status to absl::Status.
inline absl::Status ToUtilStatus(const ::tensorflow::Status& s) {
  return s.ok()
             ? absl::OkStatus()
             : absl::UnknownError(absl::StrCat("TensorFlow: ", s.ToString()));
}

// Converts absl::Status to tensorflow::Status.
inline ::tensorflow::Status FromUtilStatus(const absl::Status& s) {
  return s.ok()
             ? ::tensorflow::Status::OK()
             : ::tensorflow::Status(tensorflow::error::Code::UNKNOWN,
                                    absl::StrCat("TensorFlow: ", s.ToString()));
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

// Evaluates an expression returning a absl::Status. Returns with the TF::Status
// if the status is not "OK".
//
// Usage example:
//   tensorflow::Status f() {
//     auto g = []() -> absl::Status { ... };
//     TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(g());
//     return tensorflow::Status::OK();
//   }
#ifndef TF_RETURN_IF_ERROR_FROM_ABSL_STATUS
#define TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(expr) \
  TF_RETURN_IF_ERROR(::yggdrasil_decision_forests::utils::FromUtilStatus(expr))
#endif

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_TENSORFLOW_H_
