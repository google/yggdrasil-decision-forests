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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"

#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {
namespace {

// Converts a buffer of integer values from one precision to the other.
// The destination buffer should be already allocated .
template <typename SrcValue, typename DstValue>
void ConvertIntegerBuffer(const char* const src_buffer, size_t num_values,
                          char* dst_buffer) {
  auto* source = reinterpret_cast<const SrcValue*>(src_buffer);
  auto* destination = reinterpret_cast<DstValue*>(dst_buffer);
  std::copy(source, source + num_values, destination);
}

// Similar as "ConvertIntegerBuffer", but with the input format being an
// argument instead of a template value.
template <typename DstValue>
absl::Status ConvertIntegerBuffer(const char* const src_buffer,
                                  uint8_t num_bytes_per_source_value,
                                  size_t num_values, char* dst_buffer) {
  // Convert to the user requested precision.
  if (num_bytes_per_source_value == 1) {
    ConvertIntegerBuffer<int8_t, DstValue>(src_buffer, num_values, dst_buffer);
  } else if (num_bytes_per_source_value == 2) {
    ConvertIntegerBuffer<int16_t, DstValue>(src_buffer, num_values, dst_buffer);
  } else if (num_bytes_per_source_value == 4) {
    ConvertIntegerBuffer<int32_t, DstValue>(src_buffer, num_values, dst_buffer);
  } else if (num_bytes_per_source_value == 8) {
    ConvertIntegerBuffer<int64_t, DstValue>(src_buffer, num_values, dst_buffer);
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Non supported precision: ", num_bytes_per_source_value, " byte(s)"));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status PrepareOutputFile(absl::string_view path) {
  return absl::OkStatus();
}

absl::Status FinalizeOutputFile(absl::string_view path) {
  return absl::OkStatus();
}

std::string ShardFilename(const absl::string_view base, const int shard_idx,
                          const int num_shards) {
  return absl::StrFormat("%s_%05d-of-%05d", base, shard_idx, num_shards);
}

int NumBytes(uint64_t max_value) {
  if (max_value < 0x80) {
    return 1;
  }
  if (max_value < 0x8000) {
    return 2;
  }
  if (max_value < 0x80000000) {
    return 4;
  }
  return 8;
}

absl::Status IntegerColumnWriter::Open(absl::string_view path,
                                       int64_t max_value) {
  num_bytes_ = NumBytes(max_value);
  max_value_ = max_value;
  path_ = std::string(path);
  RETURN_IF_ERROR(PrepareOutputFile(path));
  return file_.Open(path);
}

template <typename Value>
absl::Status IntegerColumnWriter::WriteValues(absl::Span<const Value> values) {
  if (num_bytes_ == 1) {
    if constexpr (std::is_same<Value, int8_t>::value) {
      return file_.Write(
          absl::string_view(reinterpret_cast<const char*>(values.data()),
                            sizeof(Value) * values.size()));
    } else {
      return WriteValuesWithCast<Value, int8_t>(values);
    }
  } else if (num_bytes_ == 2) {
    if constexpr (std::is_same<Value, int16_t>::value) {
      return file_.Write(
          absl::string_view(reinterpret_cast<const char*>(values.data()),
                            sizeof(Value) * values.size()));
    } else {
      return WriteValuesWithCast<Value, int16_t>(values);
    }
  } else if (num_bytes_ == 4) {
    if constexpr (std::is_same<Value, int32_t>::value) {
      return file_.Write(
          absl::string_view(reinterpret_cast<const char*>(values.data()),
                            sizeof(Value) * values.size()));
    } else {
      return WriteValuesWithCast<Value, int32_t>(values);
    }
  } else if (num_bytes_ == 8) {
    if constexpr (std::is_same<Value, int64_t>::value) {
      return file_.Write(
          absl::string_view(reinterpret_cast<const char*>(values.data()),
                            sizeof(Value) * values.size()));
    } else {
      return WriteValuesWithCast<Value, int64_t>(values);
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Non supported precision ", num_bytes_));
  }
}

template absl::Status IntegerColumnWriter::WriteValues(
    absl::Span<const int8_t> values);
template absl::Status IntegerColumnWriter::WriteValues(
    absl::Span<const int16_t> values);
template absl::Status IntegerColumnWriter::WriteValues(
    absl::Span<const int32_t> values);
template absl::Status IntegerColumnWriter::WriteValues(
    absl::Span<const int64_t> values);

template <typename Value, typename DstValue>
absl::Status IntegerColumnWriter::WriteValuesWithCast(
    absl::Span<const Value> values) {
#ifndef NDEBUG
  for (const auto value : values) {
    DCHECK_LE(value, max_value_);
    if (max_value_ > 0) {
      DCHECK_GT(value, -max_value_);
    }
  }
#endif
  // TODO(gbm): Keep a buffer in between calls.
  std::vector<DstValue> dst_values(values.begin(), values.end());
  return file_.Write(
      absl::string_view(reinterpret_cast<const char*>(dst_values.data()),
                        sizeof(DstValue) * dst_values.size()));
}

absl::Status IntegerColumnWriter::Close() {
  RETURN_IF_ERROR(file_.Close());
  return FinalizeOutputFile(path_);
}

template <typename Value>
absl::Status IntegerColumnReader<Value>::Open(absl::string_view path,
                                              int64_t max_value,
                                              int max_num_values) {
  file_num_bytes_ = NumBytes(max_value);
  if (file_num_bytes_ > sizeof(Value)) {
    return absl::InvalidArgumentError(
        absl::Substitute("Template Value does not have enough precision ($0 "
                         "bytes) to read $1 with $2 byte precisions.",
                         sizeof(Value), path, file_num_bytes_));
  }
  same_user_and_file_precision_ = kUserNumBytes == file_num_bytes_;

  file_buffer_.resize(file_num_bytes_ * max_num_values);
  if (!same_user_and_file_precision_) {
    user_buffer_.resize(kUserNumBytes * max_num_values);
  }
  return file_.Open(path);
}

template <typename Value>
absl::Span<const Value> IntegerColumnReader<Value>::Values() {
  return values_;
}

template <typename Value>
absl::Span<const char> IntegerColumnReader<Value>::ActiveFileBuffer() {
  return absl::Span<const char>(file_buffer_.data(),
                                values_.size() * file_num_bytes_);
}

template <typename Value>
absl::Status IntegerColumnReader<Value>::Next() {
  ASSIGN_OR_RETURN(const auto read_bytes,
                   file_.ReadUpTo(file_buffer_.data(), file_buffer_.size()));
  DCHECK_EQ(read_bytes % file_num_bytes_, 0);
  const auto num_values = read_bytes / file_num_bytes_;
  if (!same_user_and_file_precision_) {
    // Convert to the user requested precision.
    RETURN_IF_ERROR(ConvertIntegerBuffer<Value>(
        file_buffer_.data(), file_num_bytes_, num_values, user_buffer_.data()));
  }

  if (same_user_and_file_precision_) {
    values_ = absl::Span<const Value>(
        reinterpret_cast<const Value*>(file_buffer_.data()), num_values);
  } else {
    values_ = absl::Span<const Value>(
        reinterpret_cast<const Value*>(user_buffer_.data()), num_values);
  }

  return absl::OkStatus();
}

template <typename Value>
absl::Status IntegerColumnReader<Value>::Close() {
  return file_.Close();
}

template class IntegerColumnReader<int8_t>;
template class IntegerColumnReader<int16_t>;
template class IntegerColumnReader<int32_t>;
template class IntegerColumnReader<int64_t>;

template <typename Value>
absl::Status ShardedIntegerColumnReader<Value>::ReadAndAppend(
    absl::string_view base_path, int64_t max_value, int begin_shard_idx,
    int end_shard_idx, std::vector<Value>* output) {
  ShardedIntegerColumnReader<Value> reader;
  RETURN_IF_ERROR(
      reader.Open(base_path,
                  /*max_value=*/max_value,
                  /*max_num_values=*/kIOBufferSizeInBytes / sizeof(Value),
                  /*begin_shard_idx=*/begin_shard_idx,
                  /*end_shard_idx=*/end_shard_idx));

  while (true) {
    CHECK_OK(reader.Next());
    const auto values = reader.Values();
    if (values.empty()) {
      break;
    }
    output->insert(output->end(), values.begin(), values.end());
  }
  return reader.Close();
}

template <typename Value>
absl::Status ShardedIntegerColumnReader<Value>::Open(
    absl::string_view base_path, int64_t max_value, int max_num_values,
    int begin_shard_idx, int end_shard_idx) {
  base_path_ = std::string(base_path);
  max_value_ = max_value;
  max_num_values_ = max_num_values;
  end_shard_idx_ = end_shard_idx;
  current_shard_idx_ = begin_shard_idx;
  if (current_shard_idx_ < end_shard_idx) {
    return sub_reader_.Open(
        ShardFilename(base_path_, current_shard_idx_, end_shard_idx),
        max_value_, max_num_values_);
  } else {
    return absl::OkStatus();
  }
}

template <typename Value>
absl::Span<const Value> ShardedIntegerColumnReader<Value>::Values() {
  return sub_reader_.Values();
}

template <typename Value>
absl::Status ShardedIntegerColumnReader<Value>::Next() {
  RETURN_IF_ERROR(sub_reader_.Next());
  if (sub_reader_.Values().empty() && current_shard_idx_ + 1 < end_shard_idx_) {
    RETURN_IF_ERROR(sub_reader_.Close());
    current_shard_idx_++;
    RETURN_IF_ERROR(sub_reader_.Open(
        ShardFilename(base_path_, current_shard_idx_, end_shard_idx_),
        max_value_, max_num_values_));
    return sub_reader_.Next();
  }
  return absl::OkStatus();
}

template <typename Value>
absl::Status ShardedIntegerColumnReader<Value>::Close() {
  return sub_reader_.Close();
}

template class ShardedIntegerColumnReader<int8_t>;
template class ShardedIntegerColumnReader<int16_t>;
template class ShardedIntegerColumnReader<int32_t>;
template class ShardedIntegerColumnReader<int64_t>;

template <typename Value>
void InMemoryIntegerColumnReaderFactory<Value>::Reserve(size_t num_values,
                                                        int64_t max_value) {
  file_buffer_.reserve(NumBytes(max_value) * num_values);
}

template <typename Value>
absl::Status InMemoryIntegerColumnReaderFactory<Value>::Load(
    absl::string_view base_path, int64_t max_value, int max_num_values,
    int begin_shard_idx, int end_shard_idx) {
  ShardedIntegerColumnReader<Value> file_reader;
  constexpr int buffer_size = kIOBufferSizeInBytes / sizeof(Value);
  RETURN_IF_ERROR(file_reader.Open(base_path, max_value, buffer_size,
                                   begin_shard_idx, end_shard_idx));

  while (true) {
    RETURN_IF_ERROR(file_reader.Next());
    auto file_buffer = file_reader.ActiveFileBuffer();
    if (file_buffer.empty()) {
      break;
    }
    // TODO: Estimate the final buffer size and pre-allocate it.

    file_buffer_.insert(file_buffer_.end(), file_buffer.begin(),
                        file_buffer.end());
  }
  file_buffer_.shrink_to_fit();

  file_num_bytes_ = file_reader.file_num_bytes();
  total_num_values_ = file_buffer_.size() / file_num_bytes_;
  same_user_and_file_precision_ = kUserNumBytes == file_num_bytes_;
  max_num_values_ = max_num_values;

  return file_reader.Close();
}

template <typename Value>
std::unique_ptr<typename InMemoryIntegerColumnReaderFactory<
    Value>::InMemoryIntegerColumnReader>
InMemoryIntegerColumnReaderFactory<Value>::CreateIterator() const {
  return absl::make_unique<
      InMemoryIntegerColumnReaderFactory<Value>::InMemoryIntegerColumnReader>(
      this);
}

template <typename Value>
std::unique_ptr<typename InMemoryIntegerColumnReaderFactory<
    Value>::InMemoryIntegerColumnReader>
InMemoryIntegerColumnReaderFactory<Value>::CreateIterator(
    size_t begin_idx, size_t end_idx) const {
  return absl::make_unique<
      InMemoryIntegerColumnReaderFactory<Value>::InMemoryIntegerColumnReader>(
      this, begin_idx, end_idx);
}

template <typename Value>
InMemoryIntegerColumnReaderFactory<Value>::InMemoryIntegerColumnReader::
    InMemoryIntegerColumnReader(
        const InMemoryIntegerColumnReaderFactory* const parent)
    : parent_(parent) {
  end_idx_ = parent->total_num_values_;
  if (!parent->same_user_and_file_precision_) {
    user_buffer_.resize(kUserNumBytes * parent->max_num_values_);
  }
}

template <typename Value>
InMemoryIntegerColumnReaderFactory<Value>::InMemoryIntegerColumnReader::
    InMemoryIntegerColumnReader(
        const InMemoryIntegerColumnReaderFactory* const parent,
        size_t begin_idx, size_t end_idx)
    : value_idx_(begin_idx), end_idx_(end_idx), parent_(parent) {
  if (!parent->same_user_and_file_precision_) {
    user_buffer_.resize(kUserNumBytes * parent->max_num_values_);
  }
}

template <typename Value>
absl::Span<const Value> InMemoryIntegerColumnReaderFactory<
    Value>::InMemoryIntegerColumnReader::Values() {
  return values_;
}

template <typename Value>
absl::Status
InMemoryIntegerColumnReaderFactory<Value>::InMemoryIntegerColumnReader::Next() {
  value_idx_ += values_.size();
  const auto num_values = std::min(
      end_idx_ - value_idx_, static_cast<size_t>(parent_->max_num_values_));

  const char* begin_file_buffer =
      parent_->file_buffer_.data() + value_idx_ * parent_->file_num_bytes_;

  if (parent_->same_user_and_file_precision_) {
    values_ = absl::Span<const Value>(
        reinterpret_cast<const Value*>(begin_file_buffer), num_values);
  } else {
    RETURN_IF_ERROR(
        ConvertIntegerBuffer<Value>(begin_file_buffer, parent_->file_num_bytes_,
                                    num_values, user_buffer_.data()));
    values_ = absl::Span<const Value>(
        reinterpret_cast<const Value*>(user_buffer_.data()), num_values);
  }
  return absl::OkStatus();
}

template <typename Value>
absl::Status InMemoryIntegerColumnReaderFactory<
    Value>::InMemoryIntegerColumnReader::Close() {
  // Nothing to do.
  return absl::OkStatus();
}

template class InMemoryIntegerColumnReaderFactory<int8_t>;
template class InMemoryIntegerColumnReaderFactory<int32_t>;
template class InMemoryIntegerColumnReaderFactory<int64_t>;

absl::Status FloatColumnWriter::Open(absl::string_view path) {
  path_ = std::string(path);
  RETURN_IF_ERROR(PrepareOutputFile(path));
  return file_.Open(path);
}

absl::Status FloatColumnWriter::WriteValues(absl::Span<const float> values) {
  return file_.Write(
      absl::string_view(reinterpret_cast<const char*>(values.data()),
                        sizeof(float) * values.size()));
}

absl::Status FloatColumnWriter::Close() {
  RETURN_IF_ERROR(file_.Close());
  return FinalizeOutputFile(path_);
}

absl::Status FloatColumnReader::Open(absl::string_view path,
                                     int max_num_values) {
  buffer_.resize(max_num_values);
  return file_.Open(path);
}

absl::Span<const float> FloatColumnReader::Values() {
  return absl::Span<const float>(buffer_.data(), num_values_);
}

absl::Status FloatColumnReader::Next() {
  ASSIGN_OR_RETURN(const auto read_bytes,
                   file_.ReadUpTo(reinterpret_cast<char*>(buffer_.data()),
                                  buffer_.size() * sizeof(float)));
  num_values_ = read_bytes / sizeof(float);
  return absl::OkStatus();
}

absl::Status FloatColumnReader::Close() { return file_.Close(); }

absl::Status ShardedFloatColumnReader::ReadAndAppend(
    absl::string_view base_path, int begin_shard_idx, int end_shard_idx,
    std::vector<float>* output) {
  ShardedFloatColumnReader reader;
  RETURN_IF_ERROR(
      reader.Open(base_path,
                  /*max_num_values=*/kIOBufferSizeInBytes / sizeof(float),
                  /*begin_shard_idx=*/begin_shard_idx,
                  /*end_shard_idx=*/end_shard_idx));

  while (true) {
    CHECK_OK(reader.Next());
    const auto values = reader.Values();
    if (values.empty()) {
      break;
    }
    output->insert(output->end(), values.begin(), values.end());
  }
  return reader.Close();
}

absl::Status ShardedFloatColumnReader::Open(absl::string_view base_path,
                                            int max_num_values,
                                            int begin_shard_idx,
                                            int end_shard_idx) {
  base_path_ = std::string(base_path);
  max_num_values_ = max_num_values;
  end_shard_idx_ = end_shard_idx;
  current_shard_idx_ = begin_shard_idx;
  if (current_shard_idx_ < end_shard_idx) {
    return sub_reader_.Open(
        ShardFilename(base_path_, current_shard_idx_, end_shard_idx),
        max_num_values_);
  } else {
    return absl::OkStatus();
  }
}

absl::Span<const float> ShardedFloatColumnReader::Values() {
  return sub_reader_.Values();
}

absl::Status ShardedFloatColumnReader::Next() {
  RETURN_IF_ERROR(sub_reader_.Next());
  if (sub_reader_.Values().empty() && current_shard_idx_ + 1 < end_shard_idx_) {
    RETURN_IF_ERROR(sub_reader_.Close());
    current_shard_idx_++;
    RETURN_IF_ERROR(sub_reader_.Open(
        ShardFilename(base_path_, current_shard_idx_, end_shard_idx_),
        max_num_values_));
    return sub_reader_.Next();
  }
  return absl::OkStatus();
}

absl::Status ShardedFloatColumnReader::Close() { return sub_reader_.Close(); }

InMemoryFloatColumnReaderFactory::InMemoryFloatColumnReader::
    InMemoryFloatColumnReader(
        const InMemoryFloatColumnReaderFactory* const parent)
    : parent_(parent) {}

absl::Span<const float>
InMemoryFloatColumnReaderFactory::InMemoryFloatColumnReader::Values() {
  return values_;
}

absl::Status
InMemoryFloatColumnReaderFactory::InMemoryFloatColumnReader::Next() {
  value_idx_ += values_.size();
  const auto num_values =
      std::min(parent_->buffer_.size() - value_idx_,
               static_cast<size_t>(parent_->max_num_values_));
  values_ =
      absl::Span<const float>(parent_->buffer_.data() + value_idx_, num_values);
  return absl::OkStatus();
}

absl::Status InMemoryFloatColumnReaderFactory::InMemoryFloatColumnReader::
    Close() {  // Nothing to do.
  return absl::OkStatus();
}

void InMemoryFloatColumnReaderFactory::Reserve(size_t num_values) {
  buffer_.reserve(num_values);
}

absl::Status InMemoryFloatColumnReaderFactory::Load(absl::string_view base_path,
                                                    int max_num_values,
                                                    int begin_shard_idx,
                                                    int end_shard_idx) {
  ShardedFloatColumnReader file_reader;
  constexpr int buffer_size = kIOBufferSizeInBytes / sizeof(float);
  RETURN_IF_ERROR(
      file_reader.Open(base_path, buffer_size, begin_shard_idx, end_shard_idx));

  while (true) {
    RETURN_IF_ERROR(file_reader.Next());
    if (file_reader.Values().empty()) {
      break;
    }
    // TODO: Estimate the final buffer size and pre-allocate it.

    buffer_.insert(buffer_.end(), file_reader.Values().begin(),
                   file_reader.Values().end());
  }
  buffer_.shrink_to_fit();
  max_num_values_ = max_num_values;
  return file_reader.Close();
}

std::unique_ptr<InMemoryFloatColumnReaderFactory::InMemoryFloatColumnReader>
InMemoryFloatColumnReaderFactory::CreateIterator() const {
  return absl::make_unique<
      InMemoryFloatColumnReaderFactory::InMemoryFloatColumnReader>(this);
}

}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
