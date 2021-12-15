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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_STATUS_MACROS_DEFAULT_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_STATUS_MACROS_DEFAULT_H_

#include "yggdrasil_decision_forests/utils/logging.h"

// Evaluates an expression returning a absl::Status. Returns with the status
// if the status is not "OK".
//
// Usage example:
//   absl::Status f() {
//     auto g = []() -> absl::Status { ... };
//     RETURN_IF_ERROR(g());
//     return absl::OKStatus();
//   }
#ifndef RETURN_IF_ERROR
#define RETURN_IF_ERROR(expr)                \
  {                                          \
    auto _status = (expr);                   \
    if (ABSL_PREDICT_FALSE(!_status.ok())) { \
      return _status;                        \
    }                                        \
  }
#endif

#define TOKEN_PASTE(x, y) x##y
#define CONCATENATE(x, y) TOKEN_PASTE(x, y)

// Evaluates an expression returning a utils::StatusOr. Returns with the status
// if the status is not "OK". Move the value to "lhs" and continue the execution
// otherwise.
//
// Usage example:
//   absl::Status f() {
//     auto g = []() -> utils::StatusOr<int> { ... };
//     ASSIGN_OR_RETURN(const auto x, g());
//     return absl::OKStatus();
//   }
//
// A third argument containing a extra error message is possible.
//
// Usage example:
//   absl::Status f() {
//     auto g = []() -> utils::StatusOr<int> { ... };
//     ASSIGN_OR_RETURN(const auto x, g(), _  << "Extra information");
//     return absl::OKStatus();
//   }

#define ASSIGN_OR_RETURN(...)                                        \
  SELECT_FOURTH_ARGUMENT_FROM_LIST(                                  \
      (__VA_ARGS__, ASSIGN_OR_RETURN_3ARGS, ASSIGN_OR_RETURN_2ARGS)) \
  (__VA_ARGS__)

// Don't allow for a class to be copied.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  TypeName& operator=(const TypeName&) = delete

#define ASSIGN_OR_RETURN_2ARGS(lhs, rexpr) \
  ASSIGN_OR_RETURN_2ARGS_IMP(lhs, rexpr,   \
                             CONCATENATE(_status_or_value, __LINE__))

#define ASSIGN_OR_RETURN_3ARGS(lhs, rexpr, message) \
  ASSIGN_OR_RETURN_3ARGS_IMP(lhs, rexpr, message,   \
                             CONCATENATE(_status_or_value, __LINE__))

#define SELECT_FOURTH_ARGUMENT(_1, _2, _3, _4, ...) _4
#define SELECT_FOURTH_ARGUMENT_FROM_LIST(args) SELECT_FOURTH_ARGUMENT args

#define ASSIGN_OR_RETURN_2ARGS_IMP(lhs, rexpr, tmpvar) \
  auto tmpvar = (rexpr);                               \
  if (ABSL_PREDICT_FALSE(!tmpvar.ok())) {              \
    return tmpvar.status();                            \
  }                                                    \
  lhs = std::move(tmpvar).value()

#define ASSIGN_OR_RETURN_3ARGS_IMP(lhs, rexpr, message, tmpvar) \
  auto tmpvar = (rexpr);                                        \
  if (ABSL_PREDICT_FALSE(!tmpvar.ok())) {                       \
    std::string _;                                              \
    LOG(WARNING) << message;                                    \
    return tmpvar.status();                                     \
  }                                                             \
  lhs = std::move(tmpvar).value()

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_STATUS_MACROS_DEFAULT_H_
