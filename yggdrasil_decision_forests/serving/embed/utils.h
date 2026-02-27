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
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"

namespace yggdrasil_decision_forests::serving::embed {

typedef std::variant<double, int64_t> DoubleOrInt64;

inline bool IsDouble(const DoubleOrInt64& v) {
  return std::holds_alternative<double>(v);
}
inline bool IsInt(const DoubleOrInt64& v) {
  return std::holds_alternative<int64_t>(v);
}
inline double AsDouble(const DoubleOrInt64& v) { return std::get<double>(v); }
inline int64_t AsInt(const DoubleOrInt64& v) { return std::get<int64_t>(v); }

bool IsJava(const proto::Options& options);

bool IsCpp(const proto::Options& options);

bool IsC(const proto::Options& options);

// Checks that a model name is valid. A model name can only contain certain
// letters depending on the language.
absl::Status CheckModelName(absl::string_view value,
                            proto::Options::LanguageCase language);
absl::Status CheckFeatureName(absl::string_view value);

// Converts any string into a C++ constant (with the "k") e.g. "HELLO_WORLD_1".
std::string StringToConstantSymbol(absl::string_view input);

// Converts any string into a Java Enum constant e.g. "HELLO_WORLD_1".
std::string StringToJavaEnumConstant(absl::string_view input);

// Converts any string into a c++ variable name e.g. "hello_world_1".
std::string StringToVariableSymbol(absl::string_view input);

// Converts any string into upper Camel Case.
// If "ensure_letter_first=true", adds a "V" character in front if the first
// character is not a letter.
std::string StringToCamelCase(absl::string_view input,
                              bool ensure_letter_first = true);

// Converts any string into lower Camel Case.
// If "ensure_letter_first=true", adds a "V" character in front if the first
// character is not a letter.
std::string StringToLowerCamelCase(absl::string_view input,
                                   bool ensure_letter_first = true);

// Converts any string into a c++ struct name e.g. "HelloWorld1".
// If "ensure_letter_first=true", adds a "V" character in front if the first
// character is not a letter.
std::string StringToStructSymbol(absl::string_view input,
                                 bool ensure_letter_first = true);

// Convert any string to a quoted string that can be used to initialize a string
// variable (event if the string contains " characters or line returns).
std::string QuoteString(absl::string_view input);

// Computes the number of bytes to encode an unsigned value. Can return 1, 2,
// or 4. For example, "MaxUnsignedValueToNumBytes" returns 2 for value=600
// (since using a single byte cannot encode a value greater than 255).
int MaxUnsignedValueToNumBytes(uint32_t value);

// Maximum signed value to encode with "bytes" bytes.
int32_t NumBytesToMaxSignedValue(int bytes);

// Maximum unsigned value to encode with "bytes" bytes.
uint32_t NumBytesToMaxUnsignedValue(int bytes);

// Same as MaxUnsignedValueToNumBytes, but for signed values.
int MaxSignedValueToNumBytes(int32_t value);

// Convert a proto dtype to the corresponding C++ class.
std::string DTypeToCppType(proto::DType::Enum value);

// Convert a proto dtype to the corresponding Java primitive type.
std::string DTypeToJavaType(proto::DType::Enum value);

// Integer representation to dtype.
proto::DType::Enum UnsignedIntegerToDtype(int bytes);

// C++ Integer representation e.g. uint16_t.
std::string UnsignedInteger(int bytes);
std::string SignedInteger(int bytes);
// Java integer representation. There are no unsigned primitive types in Java.
std::string JavaInteger(int bytes);

// Converts a DoubleOrInt64 variant into a valid C/C++ literal string.
// E.g., returns "40.5f" or "405" based on the expected type.
absl::StatusOr<std::string> FormatExampleLiteral(const DoubleOrInt64& val,
                                                 bool is_float);

// Maps the abstract storage requirements to C/C++ primitive names.
absl::StatusOr<std::string> StorageToPrimitiveType(int bytes, bool is_float,
                                                   bool is_signed);

// Computes the number of nodes (leaves and non-leaves) in a tree given the
// number of leaves. Note: The trees are binary trees.
int NumLeavesToNumNodes(int num_leaves);

// Indents every line of the string by the given number of spaces.
std::string IndentString(absl::string_view input, int num_spaces);

// Packs a vector of booleans into a vector of bytes (8 booleans per byte).
// The first boolean is the least significant bit of the first byte.
std::vector<uint8_t> PackBoolVector(const std::vector<bool>& input);

}  // namespace yggdrasil_decision_forests::serving::embed

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_UTILS_H_
