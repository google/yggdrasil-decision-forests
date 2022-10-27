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

// Utility macros for the manipulation of absl's status.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_STATUS_MACROS_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_STATUS_MACROS_H_

#include "yggdrasil_decision_forests/utils/status_macros_default.h"

// Status macros for Absl Status
// NOLINTBEGIN

#define STATUS_FATAL(arg) \
 return absl::InvalidArgumentError(arg)

#define STATUS_FATALS(arg, ...) \
 return absl::InvalidArgumentError(absl::StrCat(arg, __VA_ARGS__))

#define STATUS_CHECK(expr) \
  if (!(expr)) return absl::InvalidArgumentError("Check failed " #expr)
#define STATUS_CHECK_EQ(a, b) STATUS_CHECK(a == b)
#define STATUS_CHECK_NE(a, b) STATUS_CHECK(a != b)
#define STATUS_CHECK_GE(a, b) STATUS_CHECK(a >= b)
#define STATUS_CHECK_LE(a, b) STATUS_CHECK(a <= b)
#define STATUS_CHECK_GT(a, b) STATUS_CHECK(a > b)
#define STATUS_CHECK_LT(a, b) STATUS_CHECK(a < b)
// NOLINTEND


#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_STATUS_MACROS_H_
