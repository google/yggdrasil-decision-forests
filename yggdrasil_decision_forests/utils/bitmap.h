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

// Utility class to write and read dense arrays of integers with user defined
// precision (between 0 and 48 bits [defined in MultibitWriter::MaxNumBits()]).
//
// One of the goal of this class is to minimize the memory usage (by opposition
// to using std::vector<int64_t>).
//
// Note: Precision of 0 means that the number of configuration is 2^0 = 1 i.e.
// this is a number between 0 and 0 (inclusive).
//
// This class support the following operations / has the following mechanics:
//   - Random access reading and writing.
//   - Efficient sequential writing (faster than calling the random writing
//     sequentially).
//   - Special efficient code if the precision is 1 bit.
//   - Thread safe reading.
//   - Thread safe writing (in some cases, see documentation).
//   - Dense storage of the data e.g. 5 integers with 6 bits precision require
//     30 (rounded up to 32) bits of storage.
//   - Support non contiguous buffer storage i.e. storing 1B numbers with 3 bits
//     precision requires 375MB of memory that can be stored as 3 chunks of
//     100MB and one chunk of 75MB. This helps with memory fragmentation.
//
// These arrays are stored as :string.
//
// Naming convention:
//   multibitmap: Each integer can have between 1 and 48 bits of precision.
//   (single)bitmap: Each integer has exactly 1 bit of precision.
//
// (Single)bitmap relies on a different code from multibitmap.

#ifndef YGGDRASIL_DECISION_FORESTS_TOOL_BITMAP_H_
#define YGGDRASIL_DECISION_FORESTS_TOOL_BITMAP_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace bitmap {

// Allocate and set to 0 a (single) bitmap.
void AllocateAndZeroBitMap(const uint64_t size, std::string* bitmap);
// Get the i-th value in a bitmap.
template <typename T>
bool GetValueBit(const std::string& bitmap, const T index);
// Set to 1 the index-th value of the bitmap.
void SetValueBit(const uint64_t index, std::string* bitmap);
// Create a user readable string representation of a (single) bitmap.
std::string ToStringBit(const std::string& bitmap, const uint64_t size);

// Converts a bitmap stored in a string into a std::vector<bool>.
void BitmapToVectorBool(const std::string& bitmap, uint64_t size,
                        std::vector<bool>* bools);

// Allocate the memory of the bitmap.
void AllocateMultibitmap(const int32_t bits_by_elements, const uint64_t size,
                         std::string* bitmap);
// Get the i-th value in a multibitmap encoding each element with
// "bits_by_elements" bits.
uint64_t GetValueMultibit(const std::string& bitmap,
                          const int32_t bits_by_elements, const uint64_t index);
// Set the value of a multibitmap.
void SetValueMultibit(const uint64_t index, const uint64_t write_value,
                      const int32_t bits_by_elements, std::string* bitmap);
// Create a user readable string representation of a multibitmap.
// e.g. "5 8 12 67".
std::string ToStringMultibit(const std::string& bitmap,
                             const int32_t bits_by_elements,
                             const uint64_t size);
// Maximum allowed number of bits by item (i.e. 48).
int MaxNumBits();

// Returns the smallest index, greater or equal to "index", such that the item
// at the returned position is the first element stored in the next
// "buffertype" block. This low level function can be used for several
// "MultibitWriter"s to work on the same buffer in separate threads.
uint64_t NextAlignedIndex(const int32_t bits_by_elements, const uint64_t index);

// Internal block unit used to write multibitmaps.
typedef uint16_t multibitmap_buffertype;

// Utility class to read a sequence of bit values stored in a char*. Not thread
// safe.
class BitReader {
 public:
  typedef uint64_t BufferItem;

  // Open a bitmap containing "size" bits. "bitmap" should contain at least
  // ceil(size/8) bytes.
  inline void Open(const char* bitmap, size_t size) {
    bitmap_ = bitmap - sizeof(BufferItem);
    size_ = size;
    remaining_items_ = 0;
#ifndef NDEBUG
    num_read_ = 0;
#endif
  }

  // Reads and returns the next bit.
  inline bool Read() {
#ifndef NDEBUG
    DCHECK_LT(num_read_, size_);
    num_read_++;
#endif
    if (ABSL_PREDICT_FALSE(!remaining_items_)) {
      bitmap_ += sizeof(BufferItem);
      buffer_ = *reinterpret_cast<const BufferItem*>(bitmap_);
      remaining_items_ = sizeof(BufferItem) * 8 - 1;
      return buffer_ & 0x1;
    }
    remaining_items_--;
    buffer_ >>= 1;
    return buffer_ & 0x1;
  }

  void Finish() {
#ifndef NDEBUG
    DCHECK_EQ(num_read_, size_);
#endif
  }

 private:
  const char* bitmap_;
  BufferItem buffer_;
  int remaining_items_;
  size_t size_;
#ifndef NDEBUG
  size_t num_read_;
#endif
};

// Utility class to write a sequent of bits. Not thread safe.
class BitWriter {
 public:
  // "size" is the maximum number of elements to write.
  BitWriter(size_t size, std::string* bitmap);

  // Allocate the memory of the bitmap.
  void AllocateAndZeroBitMap();
  // Write a new value.
  void Write(bool value);
  // Finish the writing. If this function is not called, the last values
  // passed with "Write" might not be written.
  void Finish();

 private:
  // Buffer block.
  typedef uint64_t BufferType;
  // Number of elements (i.e. bits) in the buffer.
  size_t size_;
  // Output bitmap.
  std::string& bitmap_;
  // Current index in bitmap_.
  int64_t cur_ = 0;
  // Current bit index in bitmap_.
  int sub_cur_ = 0;
  // Buffer to store written value before being written in "bitmap_".
  BufferType buffer_ = 0;
#ifndef NDEBUG
  size_t num_written_ = 0;
#endif
};

// Utility class to write a multibitmap in order. This is equivalent but more
// efficient than calling "SetValueMultibit" repeatedly.
class MultibitWriter {
 public:
  // Create a writer with the writing head at the start of the buffer. If using
  // this constructor, the "Finish" function will check that "size" elements
  // have been written. Use the other constructor is you don't plan on writing
  // "size" elements.
  MultibitWriter(const int32_t bits_by_elements, const uint64_t size,
                 std::string* bitmap);
  // Create a writer with the writing head at a specific position. This writer
  // will modify the already existing items that overlap with the initial
  // writing head position. Use "NextAlignedIndex" to know which item will be
  // impacted.
  MultibitWriter(const int32_t bits_by_elements, const uint64_t size,
                 const uint64_t begin, std::string* bitmap);
  ~MultibitWriter();
  // Allocate the memory of the bitmap. Should be called before "Write" in case
  // the "bitmap" passed in the constructor was not already allocated.
  void AllocateAndZeroBitMap();
  // Write a new value.
  void Write(const uint64_t value);
  // Finish the writing. If this function is not called, the last values
  // passed with "Write" might not be written.
  void Finish();

 private:
  // Buffer block.
  typedef uint16_t buffertype;
  // Number of bits by element.
  int bits_by_elements_;
  // Number of elements.
  uint64_t size_;
  // Output multibitmap.
  std::string& bitmap_;
  // Current index in map_ (this is different from the writing index
  // since elements are not necessary encoded with 8 bits).
  int64_t cur_ = 0;
  // Currently bit index in map_.
  int sub_cur_ = 0;
  // Buffer to store written value before being written in "map_".
  uint64_t buffer_ = 0;
  // Index of the next to write element.
  uint64_t index_;
  // If true, "Finish" checks that "size_" elements have been written.
  bool check_full_write_;
  // Was "Finish" called?
  bool finish_called_ = false;
};

// A ShardedMultiBitmap is a multi-bitmap sharded in memory (i.e. stored in
// non-contiguous segments of contiguous memory). Using ShardedMultiBitmap is
// preferable to using string+MultiBitmap when the size of the object (number of
// elements * bits per elements) is large (Doing so helps TCMalloc with the
// memory management).
class ShardedMultiBitmap {
 public:
  // Define the size of the buffer and (optionally) allocate the memory.
  //
  // "max_num_element_in_shard" defines the maximum number of elements in a
  // shard. All except the last shard will contain "max_num_element_in_shard_"
  // elements.
  //
  // If "allocates_shards"==false, the memory of the shards is not allocated. In
  // this case, the memory should be allocated manually (using
  // "AllocateAndZeroShard") before any reading/writing/export operation.
  //
  // If called on a non-empty ShardedMultiBitmap (i.e. an already defined and
  // allocated "ShardedMultiBitmap"), previous values will be erased.
  void AllocateAndZero(const int32_t bits_by_elements,
                       const uint64_t num_elements,
                       const uint64_t max_num_element_in_shard,
                       const bool allocate_shards);
  // Allocate the memory of a single shard.
  void AllocateAndZeroShard(const uint64_t shard_idx);
  // Deallocate the memory of a single shard.
  void DeallocateShard(const uint64_t shard_idx);
  // Get the value of the index-th element.
  uint64_t GetValue(const uint64_t index) const;
  // Create a human readable representation of the multibitmap similar to
  // "ToStringMultibit".
  std::string ToString() const;
  // Number of shards.
  uint64_t NumShards() const;
  // Create a "MultibitWriter" on the entirety a given shard.
  std::unique_ptr<MultibitWriter> CreateWriter(const uint64_t shard_idx);
  // Return the index of the first element in the shard "shard_idx". Can be used
  // even if the shard has not been allocated.
  uint64_t BeginShardIndex(const uint64_t shard_idx) const;
  // Return the index of the last+1 element in the shard "shard_idx". Can be
  // used even if the shard has not been allocated.
  uint64_t EndShardIndex(const uint64_t shard_idx) const;
  // Number of elements in a shard. Can be used even if the shard have not been
  // allocated.
  uint64_t NumElementsInShard(const uint64_t shard_idx) const;
  // Size (in byte) of the allocated memory. Don't count the object structure
  // overhead. This value can be greater than ceil(number_of_elements *
  // bits_by_element/8) because each shard is stored individually on a round
  // number of bytes. Non-allocated shards are not counted.
  uint64_t TotalAllocatedMemoryInByte() const;
  // Export the sharded mutibitmap into a set of raw binary files and a proto
  // header file. "num_threads" is the number of threads/writer used for the
  // export. The created file are named "[base_path]_header" and
  // "[base_path]_shard_[shard idx]"
  absl::Status SaveToFile(const std::string& base_path,
                          const int num_threads) const;
  // Load a sharded mutibitmap from a set of raw binary files.
  absl::Status LoadFromFile(const std::string& base_path);

  uint64_t max_num_element_in_shard() const;

  int32_t bits_by_elements() const;

 private:
  // Size (in byte) of the shard_idx-th shard. Can be used if the shard is not
  // allocated.
  uint64_t ShardBufferSize(const uint64_t shard_idx) const;

  // Number of bits by elements.
  int32_t bits_by_elements_ = 0;
  // Number of elements (in total).
  uint64_t num_elements_ = 0;
  // Maximum number of elements in a shard. All shards expect the last one have
  // this exact number of elements.
  uint64_t max_num_element_in_shard_ = 0;
  // The multi-bitmaps of each shard.
  std::vector<std::string> shards_;
};

template <typename T>
bool GetValueBit(const std::string& bitmap, const T index) {
  const T byte_index = index / 8;
  DCHECK_GE(index, T{0});
  DCHECK_LT(byte_index, static_cast<T>(bitmap.size()));
  const auto byte_value = static_cast<uint8_t>(bitmap[byte_index]);
  return (byte_value & (1 << (index & 7))) != 0;
}

}  // namespace bitmap
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_TOOL_BITMAP_H_
