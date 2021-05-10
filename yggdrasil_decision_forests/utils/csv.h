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

// RFC 4180-compliant CSV Reader and Writer.
// See https://tools.ietf.org/html/rfc4180
//
// Usage example:
//
//   // Reader
//   auto file_handle = file::OpenInputFile(...).value();
//   Reader reader(file_handle.get());
//   std::vector<absl::string_view>* row;
//   while(reader.NextRow(&row).value()) {
//     // Do something with "row".
//   }
//   file_handle->Close();
//
//   // Writer
//   auto file_handle = file::OpenOutputFile(...).value();
//   Writer writer(output_handle.get());
//   writer.Write({"field1","field2"});
//   file_handle->Close();
//
// Highlights from RFC4180:
//  - Fields are separated by commas.
//  - Fields can be escaped with double quotes.
//  - Double quotes are escaped by using a pair of double-quotes.
//
// Supports Unix, Windows and Mac new lines for reading. Writes
// new lines in Unix ("\n") or Windows ("\r\n") format.
//
#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_CSV_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_CSV_H_

#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace csv {

enum class NewLine {
  UNIX,     // \n
  WINDOWS,  // \r\n
};

class Reader {
 public:
  // Created a CSV reader.
  // Does not take ownership of the stream. If "stream" is a file, it is up to
  // the caller to close it.
  Reader(InputByteStream* stream);

  // Reads the next row. The data pointed by "row" is valid until the next call
  // to "NextRow" or until the object is deleted. Returns false when no new rows
  // are available.
  utils::StatusOr<bool> NextRow(std::vector<absl::string_view>** fields);

 private:
  // Current character i.e. the last one read from the file.
  int CurrentChar();

  // Reads the next character.
  absl::Status ConsumeChar();

  // Consumes character(s) representing a end of line.
  absl::Status ConsumeEndOfRow();

  // Start the construction of a new row in the cache. Should be called before
  // any "*Cache" operations.
  void NewRowCache();

  // Adds a characters to field in construction .
  void AddCharacterToRowCache(char c);

  // Validate the field being constructed with "AddCharacterToCache", and start
  // a new one.
  void SubmitFieldToRowCache();

  // Finalize the creation of the row cache. After this call, "cached_fields_"
  // is constructed and ready to be sent to the user. The only valid next cache
  // operation is "NewRowCache".
  void FinalizeRowCache();

  // Non-owned input stream.
  InputByteStream* stream_;

  // Fields of the last read row. Points to "last_row_". Returned by the
  // public "NextRow" function.
  std::vector<absl::string_view> cached_fields_;

  // Size (in characters) of each elements in "cached_fields_".
  std::vector<int> cached_field_size_;

  // Last read row.
  std::string cached_row_;

  // File reading buffer.
  // The CSV file is read in chunks of "allocated_buffer_size_" bytes. Larger
  // values likely won't help, since the OS already provider reading buffers.
  constexpr static int allocated_buffer_size_ = 1024;
  char buffer_[allocated_buffer_size_];

  // Size of "buffer_" effectively containing data.
  int buffer_size_ = 0;

  // Index, in "buffer_", of the current character.
  int char_idx = 0;

  // Was the "buffer_" initialized.
  bool initialized_ = false;

  // Number of rows read so far.
  int num_rows = 0;
};

class Writer {
 public:
  // CSV writer constructor.
  //
  // Does not take ownership of "stream", if "stream" is a file, it
  // is up to the caller to close it.
  //
  // Fields are printed without escaping unless necessary: fields
  // containing comma (','), new-lines symbols ('\n' or '\r') or double-quotes
  // ('"').
  Writer(OutputByteStream* stream, NewLine newline = NewLine::UNIX);

  // Writes a row of data.
  absl::Status WriteRow(const std::vector<absl::string_view>& fields);

  // Helper. Use "WriteRow" when possible.
  absl::Status WriteRowStrings(const std::vector<std::string>& fields);

 private:
  // Non-owned input stream.
  OutputByteStream* stream_;

  // Character(s) representing a new line.
  std::string newline_;
};

}  // namespace csv
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_CSV_H_
