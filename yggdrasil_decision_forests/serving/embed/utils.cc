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

#include "yggdrasil_decision_forests/serving/embed/utils.h"

#include <cctype>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"

namespace yggdrasil_decision_forests::serving::embed {
namespace {

// Replacement of characters with a string for feature names. Note that feature
// names can only be: alpha numericals, _, and the values in kReplacements.
const absl::flat_hash_map<char, std::string> kReplacements = {
    {'<', "Lt"},
    {'>', "Gt"},
    {'.', "_"},
};

}  // namespace

absl::Status CheckModelName(absl::string_view value,
                            proto::Options::LanguageCase language) {
  switch (language) {
    case proto::Options::kCc:
      for (const char c : value) {
        if (!std::islower(c) && !std::isdigit(c) && c != '_') {
          return absl::InvalidArgumentError(absl::StrCat(
              "Invalid model name: ", value,
              ". The model name can only contain lowercase letters, "
              "numbers, and _."));
        }
      }
      break;
    case proto::Options::kJava:
      for (const char c : value) {
        if (!std::isalnum(c)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Invalid model name: ", value,
              ". The model name can only contain alphanumeric characters."));
        }
      }
      break;
    case proto::Options::LANGUAGE_NOT_SET:
      return absl::InternalError("Language not set for CheckModelName");
  }
  return absl::OkStatus();
}

absl::Status CheckFeatureName(absl::string_view value) {
  for (const char c : value) {
    if (!std::isalpha(c) && !std::isdigit(c) && c != '_') {
      if (kReplacements.contains(c)) {
        continue;
      }
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid model name: ", value,
          ". The feature names can only be alpha numericals, _, or "
          "symbols defined in kReplacements"));
    }
  }
  return absl::OkStatus();
}

// Converts any string into a snake case symbol.
std::string StringToSnakeCaseSymbol(const std::string_view input,
                                    const bool to_upper,
                                    const char prefix_char_if_digit) {
  if (input.empty()) {
    return "";
  }

  std::string result;
  result.reserve(input.size());
  bool last_char_was_separator = true;
  bool first_char = true;

  for (const char ch : input) {
    if (std::isalnum(ch)) {
      if (std::isdigit(ch) && first_char) {
        // Add a prefix if the first character is a number.
        result.push_back(prefix_char_if_digit);
      }
      // Change the case of letters.
      if (to_upper) {
        result.push_back(std::toupper(ch));
      } else {
        result.push_back(std::tolower(ch));
      }
      last_char_was_separator = false;
    } else if (ch == ' ' || ch == '-' || ch == '_') {
      // Characters that are replaced with "_".
      if (!result.empty() && !last_char_was_separator) {
        result.push_back('_');
        last_char_was_separator = true;
      }
    }

    const auto replace_it = kReplacements.find(ch);
    if (replace_it != kReplacements.end()) {
      last_char_was_separator = true;
      if (to_upper) {
        absl::StrAppend(&result, absl::AsciiStrToUpper(replace_it->second));
      } else {
        absl::StrAppend(&result, replace_it->second);
      }
    }

    // Other characters are skipped.
    first_char = false;
  }

  // Remove the last character if it is a separator.
  if (!result.empty() && result.back() == '_') {
    result.pop_back();
  }
  return result;
}

std::string StringToConstantSymbol(const absl::string_view input) {
  return StringToSnakeCaseSymbol(input, true, 'V');
}

std::string StringToJavaEnumConstant(absl::string_view input) {
  return StringToSnakeCaseSymbol(input, true, 'V');
}

std::string StringToVariableSymbol(const absl::string_view input) {
  return StringToSnakeCaseSymbol(input, false, 'v');
}

std::string StringToStructSymbol(const absl::string_view input,
                                 const bool ensure_letter_first) {
  return StringToCamelCase(input, ensure_letter_first);
}

std::string StringToLowerCamelCase(absl::string_view input,
                                   bool ensure_letter_first) {
  auto camel_case = StringToCamelCase(input, ensure_letter_first);
  if (!camel_case.empty()) {
    camel_case[0] = std::tolower(camel_case[0]);
  }
  return camel_case;
}

std::string StringToCamelCase(absl::string_view input,
                              bool ensure_letter_first) {
  if (input.empty()) {
    return "";
  }

  std::string result;
  result.reserve(input.size());
  bool capitalize_next_char = true;
  bool current_word_started_with_letter = false;
  bool first_char = true;

  for (const char ch : input) {
    if (std::isalnum(ch)) {
      if (ensure_letter_first && std::isdigit(ch) && first_char) {
        // Add a prefix if the first character is a number.
        result.push_back('V');
      }
      // Change the case of letters.
      if (capitalize_next_char) {
        result.push_back(std::toupper(ch));
        current_word_started_with_letter = std::isalpha(ch);
        capitalize_next_char = false;
      } else {
        if (current_word_started_with_letter && std::isalpha(ch)) {
          result.push_back(std::tolower(ch));
        } else {
          result.push_back(ch);
        }
      }
    } else {
      capitalize_next_char = true;

      const auto replace_it = kReplacements.find(ch);
      if (replace_it != kReplacements.end()) {
        absl::StrAppend(&result, replace_it->second);
      }
    }
    // Other characters are skipped.
    first_char = false;
  }

  // Remove the last character if it is a separator.
  if (!result.empty() && result.back() == '_') {
    result.pop_back();
  }
  return result;
}

int MaxUnsignedValueToNumBytes(uint32_t value) {
  if (value <= 0xff) {
    return 1;
  } else if (value <= 0xffff) {
    return 2;
  } else {
    return 4;
  }
}

uint32_t NumBytesToMaxUnsignedValue(int bytes) {
  switch (bytes) {
    case 1:
      return 0xff;
    case 2:
      return 0xffff;
    case 4:
      return 0xffffffff;
    default:
      DCHECK(false);
      return 0;
  }
}

int MaxSignedValueToNumBytes(int32_t value) {
  if (value <= 0x7f && value >= -0x80) {
    return 1;
  } else if (value <= 0x7fff && value >= -0x8000) {
    return 2;
  } else {
    return 4;
  }
}

std::string UnsignedInteger(int bytes) {
  return absl::StrCat("uint", bytes * 8, "_t");
}
std::string SignedInteger(int bytes) {
  return absl::StrCat("int", bytes * 8, "_t");
}
std::string JavaInteger(int bytes) {
  switch (bytes) {
    case 1:
      return "byte";
    case 2:
      return "short";
    case 4:
      return "int";
    case 8:
      return "long";
    default:
      DCHECK(false) << "Invalid number of bytes for JavaInteger: " << bytes;
      return "";
  }
}

std::string DTypeToCCType(const proto::DType::Enum value) {
  switch (value) {
    case proto::DType::UNDEFINED:
      DCHECK(false);
      return "UNDEFINED";

    case proto::DType::INT8:
      return "int8_t";
    case proto::DType::INT16:
      return "int16_t";
    case proto::DType::INT32:
      return "int32_t";

    case proto::DType::UINT8:
      return "uint8_t";
    case proto::DType::UINT16:
      return "uint16_t";
    case proto::DType::UINT32:
      return "uint32_t";

    case proto::DType::FLOAT32:
      return "float";

    case proto::DType::BOOL:
      return "bool";
  }
}

proto::DType::Enum UnsignedIntegerToDtype(int bytes) {
  switch (bytes) {
    case 1:
      return proto::DType::UINT8;
    case 2:
      return proto::DType::UINT16;
    case 4:
      return proto::DType::UINT32;
    default:
      DCHECK(false);
      return proto::DType::UNDEFINED;
  }
}

int NumLeavesToNumNodes(int num_leaves) {
  DCHECK_GE(num_leaves, 0);
  if (num_leaves == 0) {
    return 0;
  }
  return 2 * num_leaves - 1;
}

std::string QuoteString(absl::string_view input) {
  return absl::StrCat("\"", absl::CEscape(input), "\"");
}

}  // namespace yggdrasil_decision_forests::serving::embed
