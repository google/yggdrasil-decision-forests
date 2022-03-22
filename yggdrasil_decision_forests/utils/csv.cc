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

#include "yggdrasil_decision_forests/utils/csv.h"

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace csv {
namespace {

// States of the parsing state machine.
enum State {
  START,
  NON_QUOTED_FIELD,
  QUOTED_FIELD,
  SECOND_QUOTE_IN_QUOTED_FIELD,
  END
};

// Tests if the character represents the end of a row i.e. new line or
// end-of-file.
bool IsEndOfRow(int c) { return c == '\n' || c == '\r' || c == /*EOF*/ -1; }

// Tests if a character needs escaping.
bool NeedsEscape(char c) {
  return c == '\n' || c == '\r' || c == ',' || c == '\"';
}

// Tests if a field needs escaping.
bool NeedsEscape(absl::string_view field) {
  for (const char c : field) {
    if (NeedsEscape(c)) {
      return true;
    }
  }
  return false;
}

}  // namespace

Reader::Reader(InputByteStream* stream) : stream_(stream) {}

utils::StatusOr<bool> Reader::NextRow(std::vector<absl::string_view>** fields) {
  // Initialize returned value.
  *fields = &cached_fields_;

  // Start recording a new row.
  NewRowCache();

  if (!initialized_) {
    initialized_ = true;
    // Read the first character of the file.
    RETURN_IF_ERROR(ConsumeChar());
  }

  if (CurrentChar() == -1) {
    // The end of file is reached.
    return false;
  }

  // Current state.
  State state = START;

  // Consumes the current characters and change the state.
  const auto set_state_and_consume = [&](State new_state) -> absl::Status {
    state = new_state;
    return ConsumeChar();
  };

  // Run the automate.
  while (state != END) {
    switch (state) {
      case START:
        if (CurrentChar() == '"') {
          RETURN_IF_ERROR(set_state_and_consume(QUOTED_FIELD));
        } else if (IsEndOfRow(CurrentChar())) {
          SubmitFieldToRowCache();
          RETURN_IF_ERROR(ConsumeEndOfRow());
          state = END;
        } else if (CurrentChar() == ',') {
          SubmitFieldToRowCache();
          RETURN_IF_ERROR(ConsumeChar());
        } else {
          AddCharacterToRowCache(CurrentChar());
          RETURN_IF_ERROR(set_state_and_consume(NON_QUOTED_FIELD));
        }
        break;

      case NON_QUOTED_FIELD:
        if (CurrentChar() == ',') {
          SubmitFieldToRowCache();
          RETURN_IF_ERROR(set_state_and_consume(START));
        } else if (CurrentChar() == '"') {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Quote in non quoted field at line %d", num_rows));
        } else if (IsEndOfRow(CurrentChar())) {
          SubmitFieldToRowCache();
          RETURN_IF_ERROR(ConsumeEndOfRow());
          state = END;
        } else {
          AddCharacterToRowCache(CurrentChar());
          RETURN_IF_ERROR(ConsumeChar());
        }
        break;

      case QUOTED_FIELD:
        if (CurrentChar() == '"') {
          RETURN_IF_ERROR(set_state_and_consume(SECOND_QUOTE_IN_QUOTED_FIELD));
        } else if (CurrentChar() == -1) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "End of file reached in a quote at line %d", num_rows));
        } else {
          AddCharacterToRowCache(CurrentChar());
          RETURN_IF_ERROR(ConsumeChar());
        }
        break;

      case SECOND_QUOTE_IN_QUOTED_FIELD:
        if (CurrentChar() == '"') {
          AddCharacterToRowCache(CurrentChar());
          RETURN_IF_ERROR(set_state_and_consume(QUOTED_FIELD));
        } else if (IsEndOfRow(CurrentChar())) {
          SubmitFieldToRowCache();
          RETURN_IF_ERROR(ConsumeEndOfRow());
          state = END;
        } else if (CurrentChar() == ',') {
          SubmitFieldToRowCache();
          RETURN_IF_ERROR(set_state_and_consume(START));
        } else {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Unexpected character after quote: '%c' at line %d",
              CurrentChar(), num_rows));
        }
        break;

      case END:
        break;
    }
  }
  FinalizeRowCache();
  return true;
}

absl::Status Reader::ConsumeEndOfRow() {
  if (!IsEndOfRow(CurrentChar())) {
    return absl::InternalError(absl::StrFormat(
        "The current character is not an end of row '%c' at line %d",
        CurrentChar(), num_rows));
  }
  if (CurrentChar() == '\r') {
    RETURN_IF_ERROR(ConsumeChar());
  }
  if (CurrentChar() == '\n') {
    RETURN_IF_ERROR(ConsumeChar());
  }
  return absl::OkStatus();
}

int Reader::CurrentChar() {
  if (char_idx < buffer_size_) {
    return buffer_[char_idx];
  } else {
    return -1;
  }
}

absl::Status Reader::ConsumeChar() {
  char_idx++;
  if (char_idx >= buffer_size_) {
    char_idx = 0;
    ASSIGN_OR_RETURN(buffer_size_, stream_->ReadUpTo(buffer_, sizeof(buffer_)));
  }
  return absl::OkStatus();
}

void Reader::NewRowCache() {
  num_rows += 1;
  cached_row_.clear();
  cached_fields_.clear();
  cached_field_size_.assign(1, 0);
}

void Reader::AddCharacterToRowCache(char c) { cached_row_.push_back(c); }

void Reader::SubmitFieldToRowCache() {
  cached_field_size_.push_back(cached_row_.size());
}

void Reader::FinalizeRowCache() {
  const auto n = cached_field_size_.size() - 1;
  cached_fields_.resize(n);
  for (size_t i = 0; i < n; i++) {
    cached_fields_[i] =
        absl::string_view(&cached_row_[cached_field_size_[i]],
                          cached_field_size_[i + 1] - cached_field_size_[i]);
  }
}

Writer::Writer(OutputByteStream* stream, NewLine newline) : stream_(stream) {
  switch (newline) {
    case NewLine::UNIX:
      newline_ = "\n";
      break;
    case NewLine::WINDOWS:
      newline_ = "\r\n";
      break;
  }
}

absl::Status Writer::WriteRow(const std::vector<absl::string_view>& fields) {
  for (int field_idx = 0; field_idx < fields.size(); field_idx++) {
    const auto& field = fields[field_idx];
    if (field_idx > 0) {
      RETURN_IF_ERROR(stream_->Write(","));
    }
    if (NeedsEscape(field)) {
      RETURN_IF_ERROR(stream_->Write("\""));
      RETURN_IF_ERROR(
          stream_->Write(absl::StrReplaceAll(field, {{"\"", "\"\""}})));
      RETURN_IF_ERROR(stream_->Write("\""));
    } else {
      RETURN_IF_ERROR(stream_->Write(field));
    }
  }
  RETURN_IF_ERROR(stream_->Write(newline_));
  return absl::OkStatus();
}

absl::Status Writer::WriteRowStrings(const std::vector<std::string>& fields) {
  return WriteRow({fields.begin(), fields.end()});
}

}  // namespace csv
}  // namespace utils
}  // namespace yggdrasil_decision_forests
