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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_UTILS_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_UTILS_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"

namespace yggdrasil_decision_forests::serving::embed {

// Checks that a model name is valid. A model name can only contain letters,
// numbers, and _.
absl::Status CheckModelName(absl::string_view value);

// Converts any string into a c++ constant (with the "k") e.g. "HELLO_WOLRD_1".
std::string StringToConstantSymbol(absl::string_view input);

// Converts any string into a c++ variable name e.g. "hello_world_1".
std::string StringToVariableSymbol(absl::string_view input);

// Converts any string into a c++ struct name e.g. "HelloWorld1".
std::string StringToStructSymbol(absl::string_view input);

// Computes the number of bytes to encode an unsigned value. Can return 1, 2,
// or 4. For example, "MaxUnsignedValueToNumBytes" returns 2 for value=600
// (since using a single byte cannot encode a value greater than 255).
int MaxUnsignedValueToNumBytes(uint32_t value);

// Same as MaxUnsignedValueToNumBytes, but for signed values.
int MaxSignedValueToNumBytes(int32_t value);

// Convert a proto dtype to the corresponding c++ class.
std::string DTypeToCCType(proto::DType::Enum value);

// Integer representation e.g. uint16_t.
std::string UnsignedInteger(int bytes);
std::string SignedInteger(int bytes);

}  // namespace yggdrasil_decision_forests::serving::embed

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_UTILS_H_
