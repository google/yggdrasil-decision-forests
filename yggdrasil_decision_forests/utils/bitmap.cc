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

#include "yggdrasil_decision_forests/utils/bitmap.h"

#include <stddef.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/utils/bitmap.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace bitmap {

namespace {
constexpr char kShardedMultiBitmapFileHeaderSuffix[] = "_header";
constexpr char kShardedMultiBitmapFileShardSuffix[] = "_shard_";

// Always gets an unsigned value from a std::string.
template <typename T>
uint8_t SafeGet(const std::string& bitmap, T index) {
  return bitmap[index];
}

template <typename T>
uint8_t& SafeGetRef(std::string& bitmap, T index) {
  return *reinterpret_cast<uint8_t*>(&bitmap[index]);
}

template <typename BufferType, typename T>
BufferType& SafeGetRefBufferType(std::string& bitmap, T index) {
  return *reinterpret_cast<BufferType*>(&bitmap[index]);
}

template <typename T>
uint8_t& SafeGetPtrRef(std::string* bitmap, T index) {
  return *reinterpret_cast<uint8_t*>(&(*bitmap)[index]);
}

}  // namespace

void AllocateAndZeroBitMap(const uint64_t size, std::string* bitmap) {
  bitmap->assign((size + 7) / 8, 0);
}

void SetValueBit(const uint64_t index, std::string* bitmap) {
  const int64_t byte_index = index / 8;
  DCHECK_GE(index, 0);
  DCHECK_LT(byte_index, bitmap->size());
  auto& byte_value = SafeGetPtrRef(bitmap, byte_index);
  byte_value = byte_value | (1 << (index & 7));
}

std::string ToStringBit(const std::string& bitmap, const uint64_t size) {
  std::string result;
  for (int64_t idx = 0; idx < size; idx++) {
    absl::StrAppend(&result, GetValueBit(bitmap, idx) ? "1" : "0");
  }
  return result;
}

void BitmapToVectorBool(const std::string& bitmap, const uint64_t size,
                        std::vector<bool>* bools) {
  bools->resize(size);
  for (int64_t idx = 0; idx < size; idx++) {
    (*bools)[idx] = GetValueBit(bitmap, idx);
  }
}

uint64_t GetValueMultibit(const std::string& bitmap,
                          const int32_t bits_by_elements,
                          const uint64_t index) {
  if (bits_by_elements == 0) return 0;
  const int64_t begin_byte_index = index * bits_by_elements / 8;
  const int64_t num_bytes =
      ((index + 1) * bits_by_elements - 1) / 8 + 1 - begin_byte_index;
  const int64_t bit_offset = index * bits_by_elements - begin_byte_index * 8;
  DCHECK_GE(bit_offset, 0);
  DCHECK_LT(bit_offset, 8);
  DCHECK_LT(begin_byte_index, bitmap.size());
  DCHECK_GE(begin_byte_index, 0);

  uint64_t buffer =
      static_cast<uint64_t>(SafeGet(bitmap, begin_byte_index)) >> bit_offset;

  for (int byte_index = 1; byte_index < num_bytes; byte_index++) {
    DCHECK_LT(byte_index + begin_byte_index, bitmap.size());
    buffer |=
        static_cast<uint64_t>(SafeGet(bitmap, byte_index + begin_byte_index))
        << (byte_index * 8 - bit_offset);
  }

  buffer &= (static_cast<uint64_t>(1) << bits_by_elements) - 1;

  return buffer;
}

void SetValueMultibit(const uint64_t index, const uint64_t write_value,
                      const int32_t bits_by_elements, std::string* bitmap) {
  DCHECK_LT(write_value, static_cast<uint64_t>(1) << bits_by_elements);
  uint64_t value = write_value;
  const uint64_t begin_byte_idx = (index * bits_by_elements) / 8;
  const uint64_t end_byte_idx = ((index + 1) * bits_by_elements + 7) / 8;

  const int begin_bit_shift = (index * bits_by_elements) - begin_byte_idx * 8;
  const int end_bit_shit =
      (index + 1) * bits_by_elements - ((index + 1) * bits_by_elements / 8) * 8;
  // Writing of a single bit.
  if (begin_byte_idx + 1 == end_byte_idx) {
    uint8_t new_mask = ((1 << bits_by_elements) - 1) << begin_bit_shift;
    uint8_t old_mask = 0xFF ^ new_mask;
    auto& v = SafeGetPtrRef(bitmap, begin_byte_idx);
    v = (v & old_mask) | (value << begin_bit_shift);
    return;
  }
  // Begin block.
  {
    auto& v = SafeGetPtrRef(bitmap, begin_byte_idx);
    const uint8_t previous_value_mask = (1 << begin_bit_shift) - 1;
    const int consumed_num_bits = 8 - begin_bit_shift;
    const uint8_t new_value_part = value & ((1 << consumed_num_bits) - 1);
    v = (v & previous_value_mask) | (new_value_part << begin_bit_shift);
    value >>= consumed_num_bits;
  }
  // Middle block.
  const uint64_t middle_end_byte_idx =
      (end_bit_shit == 0) ? (end_byte_idx) : (end_byte_idx - 1);
  for (uint64_t cur_byte = begin_byte_idx + 1; cur_byte < middle_end_byte_idx;
       cur_byte++) {
    auto& v = SafeGetPtrRef(bitmap, cur_byte);
    v = value & 0xFF;
    value >>= 8;
  }
  // End block.
  if (end_bit_shit != 0) {
    auto& v = SafeGetPtrRef(bitmap, end_byte_idx - 1);
    const uint8_t new_value_mask = (1 << end_bit_shit) - 1;
    const uint8_t previous_value_mask = 0xFF ^ new_value_mask;
    v = (v & previous_value_mask) | (value & new_value_mask);
  }
}

std::string ToStringMultibit(const std::string& bitmap,
                             const int32_t bits_by_elements,
                             const uint64_t size) {
  std::string result;
  for (int64_t i = 0; i < size; i++) {
    if (i > 0) absl::StrAppend(&result, " ");
    absl::StrAppend(&result, GetValueMultibit(bitmap, bits_by_elements, i));
  }
  return result;
}

int MaxNumBits() {
  return (sizeof(uint64_t) - sizeof(multibitmap_buffertype)) * 8 - 1;
}

void AllocateMultibitmap(const int32_t bits_by_elements, const uint64_t size,
                         std::string* bitmap) {
  bitmap->assign((size * bits_by_elements + 7) / 8, 0);
}

uint64_t NextAlignedIndex(const int32_t bits_by_elements,
                          const uint64_t index) {
  if (bits_by_elements == 0) return index;
  const int64_t block_idx =
      (index * bits_by_elements + sizeof(multibitmap_buffertype) * 8 - 1) /
      (sizeof(multibitmap_buffertype) * 8);
  return (block_idx * sizeof(multibitmap_buffertype) * 8 + bits_by_elements -
          1) /
         bits_by_elements;
}

BitWriter::BitWriter(const size_t size, std::string* bitmap)
    : size_(size), bitmap_(*bitmap) {}

void BitWriter::AllocateAndZeroBitMap() { bitmap_.resize((size_ + 7) / 8); }

void BitWriter::Write(bool value) {
#ifndef NDEBUG
  DCHECK_LT(num_written_, size_);
  num_written_++;
#endif
  buffer_ |= static_cast<uint64_t>(value) << sub_cur_;
  if ((sub_cur_++) == sizeof(BufferType) * 8 - 1) {
    DCHECK_LT(cur_, bitmap_.size());
    SafeGetRefBufferType<BufferType>(bitmap_, cur_) = buffer_;
    sub_cur_ = 0;
    cur_ += sizeof(BufferType);
    buffer_ = 0;
  }
}

void BitWriter::Finish() {
#ifndef NDEBUG
  DCHECK_EQ(num_written_, size_);
#endif
  if (sub_cur_ > 0 && sub_cur_ < sizeof(BufferType) * 8) {
    const int num_tails = (sub_cur_ + 7) / 8;
    for (int tail = 0; tail < num_tails; tail++) {
      const uint8_t tailvalue = buffer_ & 0xFF;
      buffer_ >>= 8;
      SafeGetRef(bitmap_, cur_ + tail) = tailvalue;
    }
  }
}

MultibitWriter::MultibitWriter(const int32_t bits_by_elements,
                               const uint64_t size, std::string* bitmap)
    : bits_by_elements_(bits_by_elements),
      size_(size),
      bitmap_(*bitmap),
      index_(0),
      check_full_write_(true) {}

MultibitWriter::MultibitWriter(const int32_t bits_by_elements,
                               const uint64_t size, const uint64_t begin,
                               std::string* bitmap)
    : bits_by_elements_(bits_by_elements),
      size_(size),
      bitmap_(*bitmap),
      index_(begin),
      check_full_write_(false) {
  DCHECK_LE(begin, size);
  const int64_t num_bits = begin * bits_by_elements;
  cur_ = num_bits / 8;
  sub_cur_ = num_bits - cur_ * 8;
  if (cur_ < bitmap_.size()) {
    buffer_ = SafeGet(bitmap_, cur_) & ((1 << sub_cur_) - 1);
  }
}

MultibitWriter::~MultibitWriter() { CHECK(finish_called_); }

void MultibitWriter::AllocateAndZeroBitMap() {
  AllocateMultibitmap(bits_by_elements_, size_, &bitmap_);
}

void MultibitWriter::Write(const uint64_t value) {
  DCHECK_LT(value, (static_cast<uint64_t>(1) << bits_by_elements_));
  DCHECK_LT(index_, size_);
  DCHECK(!finish_called_);
  index_++;
  buffer_ |= value << sub_cur_;
  sub_cur_ += bits_by_elements_;
  while (sub_cur_ >= sizeof(buffertype) * 8) {
    SafeGetRefBufferType<buffertype>(bitmap_, cur_) =
        (buffer_ & std::numeric_limits<buffertype>::max());
    cur_ += sizeof(buffertype);
    sub_cur_ -= sizeof(buffertype) * 8;
    buffer_ >>= sizeof(buffertype) * 8;
  }
}

void MultibitWriter::Finish() {
  CHECK(!finish_called_);
  CHECK(!check_full_write_ || index_ == size_);
  finish_called_ = true;
  if (sub_cur_ > 0) {
    const int num_tails = (sub_cur_ + 7) / 8;
    const int num_bits_in_last_tail = sub_cur_ % 8;
    for (int tail = 0; tail < num_tails; tail++) {
      auto tailvalue = static_cast<uint8_t>(buffer_ & 0xFF);
      buffer_ >>= 8;
      auto& dst_byte = SafeGetRef(bitmap_, cur_ + tail);
      if (tail == num_tails - 1 && num_bits_in_last_tail != 0) {
        // Adds to "tailvalue" the bits of the next items in the buffer (if any)
        // so they are not overridden by "dst_byte = tailvalue".
        tailvalue |= dst_byte & (~((1 << num_bits_in_last_tail) - 1));
      }
      dst_byte = tailvalue;
    }
  }
}

void ShardedMultiBitmap::AllocateAndZero(
    const int32_t bits_by_elements, const uint64_t num_elements,
    const uint64_t max_num_element_in_shard, const bool allocate_shards) {
  CHECK_GT(max_num_element_in_shard, 0);
  bits_by_elements_ = bits_by_elements;
  num_elements_ = num_elements;
  max_num_element_in_shard_ = max_num_element_in_shard;
  const uint64_t num_shards = (num_elements_ + max_num_element_in_shard_ - 1) /
                              max_num_element_in_shard_;
  shards_.assign(num_shards, std::string());
  if (allocate_shards) {
    for (uint64_t shard_idx = 0; shard_idx < num_shards; shard_idx++) {
      AllocateAndZeroShard(shard_idx);
    }
  }
}

void ShardedMultiBitmap::AllocateAndZeroShard(const uint64_t shard_idx) {
  auto buffer_size = ShardBufferSize(shard_idx);
  shards_[shard_idx].assign(buffer_size, 0);
}

void ShardedMultiBitmap::DeallocateShard(const uint64_t shard_idx) {
  shards_[shard_idx].clear();
  shards_[shard_idx].shrink_to_fit();
}

uint64_t ShardedMultiBitmap::GetValue(const uint64_t index) const {
  const uint64_t shard_idx = index / max_num_element_in_shard_;
  const size_t idx_in_shard = index - shard_idx * max_num_element_in_shard_;
  DCHECK_GE(shard_idx, 0);
  DCHECK_LT(shard_idx, shards_.size());
  DCHECK(!shards_[shard_idx].empty());
  return GetValueMultibit(shards_[shard_idx], bits_by_elements_, idx_in_shard);
}

std::string ShardedMultiBitmap::ToString() const {
  std::string result;
  for (uint64_t idx = 0; idx < num_elements_; idx++) {
    if (idx > 0) absl::StrAppend(&result, " ");
    absl::StrAppend(&result, GetValue(idx));
  }
  return result;
}

uint64_t ShardedMultiBitmap::NumShards() const { return shards_.size(); }

std::unique_ptr<MultibitWriter> ShardedMultiBitmap::CreateWriter(
    const uint64_t shard_idx) {
  DCHECK_GE(shard_idx, 0);
  DCHECK_LT(shard_idx, shards_.size());
  DCHECK(!shards_[shard_idx].empty());
  return absl::make_unique<MultibitWriter>(
      bits_by_elements_, NumElementsInShard(shard_idx), &shards_[shard_idx]);
}

uint64_t ShardedMultiBitmap::BeginShardIndex(const uint64_t shard_idx) const {
  DCHECK_GE(shard_idx, 0);
  DCHECK_LT(shard_idx, shards_.size());
  return max_num_element_in_shard_ * shard_idx;
}

uint64_t ShardedMultiBitmap::EndShardIndex(const uint64_t shard_idx) const {
  DCHECK_GE(shard_idx, 0);
  DCHECK_LT(shard_idx, shards_.size());
  return std::min(max_num_element_in_shard_ * (shard_idx + 1), num_elements_);
}

uint64_t ShardedMultiBitmap::NumElementsInShard(
    const uint64_t shard_idx) const {
  return EndShardIndex(shard_idx) - BeginShardIndex(shard_idx);
}

uint64_t ShardedMultiBitmap::max_num_element_in_shard() const {
  return max_num_element_in_shard_;
}

int32_t ShardedMultiBitmap::bits_by_elements() const {
  return bits_by_elements_;
}

uint64_t ShardedMultiBitmap::TotalAllocatedMemoryInByte() const {
  uint64_t size = 0;
  for (const auto& shard : shards_) {
    size += shard.size();
  }
  return size;
}

uint64_t ShardedMultiBitmap::ShardBufferSize(const uint64_t shard_idx) const {
  uint64_t num_elements = NumElementsInShard(shard_idx);
  return (num_elements * bits_by_elements_ + 8 - 1) / 8;
}

absl::Status ShardedMultiBitmap::SaveToFile(const std::string& base_path,
                                            const int num_threads) const {
  proto::ShardedMultiBitmapHeader header;
  header.set_bits_by_elements(bits_by_elements_);
  header.set_num_elements(num_elements_);
  header.set_max_num_element_in_shard(max_num_element_in_shard_);
  header.set_num_shards(shards_.size());
  RETURN_IF_ERROR(file::SetBinaryProto(
      absl::StrCat(base_path, kShardedMultiBitmapFileHeaderSuffix), header,
      file::Defaults()));
  absl::Status status;
  utils::concurrency::Mutex status_mutex;
  {
    yggdrasil_decision_forests::utils::concurrency::ThreadPool pool(
        "ShardedMultiBitmap::SaveToFile", num_threads);
    pool.StartWorkers();
    for (uint64_t shard_idx = 0; shard_idx < shards_.size(); shard_idx++) {
      pool.Schedule([shard_idx, base_path, &status, &status_mutex, this]() {
        auto local_status = file::SetContent(
            absl::StrCat(base_path, kShardedMultiBitmapFileShardSuffix,
                         shard_idx),
            shards_[shard_idx]);
        if (!local_status.ok()) {
          utils::concurrency::MutexLock lock(&status_mutex);
          status.Update(local_status);
        }
      });
    }
  }
  return status;
}

absl::Status ShardedMultiBitmap::LoadFromFile(const std::string& base_path) {
  proto::ShardedMultiBitmapHeader header;
  RETURN_IF_ERROR(file::GetBinaryProto(
      absl::StrCat(base_path, kShardedMultiBitmapFileHeaderSuffix), &header,
      file::Defaults()));
  AllocateAndZero(header.bits_by_elements(), header.num_elements(),
                  header.max_num_element_in_shard(), false);
  for (uint64_t shard_idx = 0; shard_idx < shards_.size(); shard_idx++) {
    ASSIGN_OR_RETURN(
        shards_[shard_idx],
        file::GetContent(absl::StrCat(
            base_path, kShardedMultiBitmapFileShardSuffix, shard_idx)));
  }
  return absl::OkStatus();
}

}  // namespace bitmap
}  // namespace utils
}  // namespace yggdrasil_decision_forests
