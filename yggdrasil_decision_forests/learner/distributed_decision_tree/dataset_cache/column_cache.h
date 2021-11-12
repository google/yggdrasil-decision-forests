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

// Utility to read and write sequences of float and integer values from / to
// disk, as well as to hold them in memory with a minimal amount of ram usage.
//
// The file format is designed for temporary storage of a given process.
// Notably, it should not be used as a way to store datasets:
//   - The file format is dependent of the host Endianness.
//   - The format of the file encoding might change without retro compatibility
//     guarantees.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_COLUMN_CACHE_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_COLUMN_CACHE_H_

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {

// Recommended size of buffers for IO operations (in bytes).
constexpr int kIOBufferSizeInBytes = 1 << 20;  // 1MB

// Number of bytes requires to store a signed integer value in [-max_value-1,
// max_value]. The returned value is one of the following: 1, 2, 4, 8.
int NumBytes(uint64_t max_value);

// Path to a file in a sharded set of files. Output
// "{base}_{shard_idx}-of-{num_shards}".
std::string ShardFilename(absl::string_view base, int shard_idx,
                          const int num_shards);

// Writes a sequence of integers.
class IntegerColumnWriter {
 public:
  // Opens a file. All the written values should be in [-max_value-1,
  // max_value].
  //
  // The behavior is undefined for values outside of this range (DCHECK in debug
  // mode).
  absl::Status Open(absl::string_view path, int64_t max_value);

  // Writes a sequence of values.
  template <typename Value>
  absl::Status WriteValues(absl::Span<const Value> values);

  // Close the file.
  absl::Status Close();

 private:
  // Writes a set of values with a precision known at compile time.
  template <typename Value, typename DstValue>
  absl::Status WriteValuesWithCast(absl::Span<const Value> values);

  // Written values should be in [-max_value-1, max_value].
  int64_t max_value_;

  // Current open file.
  file::FileOutputByteStream file_;

  // Number of bytes used by each value.
  uint8_t num_bytes_ = 0;

  // Open file.
  std::string path_;
};

// Abstract class to reads a sequence of integer.
template <typename Value>
class AbstractIntegerColumnIterator {
 public:
  virtual ~AbstractIntegerColumnIterator() = default;

  // Reads a sequence of values and make them available in "Values()". "Next"
  // should be called before the first call to "Values".
  virtual absl::Status Next() = 0;

  // Returns the latest read values.
  //
  // The returned values will be invalidated at the next call to "Next" or at
  // the destruction / closing of the reader object. If the number of returned
  // values is empty, the stream is done being read. The number of returned
  // values at depends on the specific implementation but will be constant
  // (except for the last set of values).
  virtual absl::Span<const Value> Values() = 0;

  // Stops the iteration. There should not be any call to "Values" or "Next"
  // after "Close".
  virtual absl::Status Close() = 0;
};

// Reads a sequence of "Value" integer stored in file.
//
// The precision of "Value" (template argument) does not have to be the same as
// the precision of "Value" in the writer. However the value of the "max_value"
// argument should be the same.
template <typename Value>
class IntegerColumnReader : public AbstractIntegerColumnIterator<Value> {
 public:
  // Number of bytes taken by one value stored in the user format.
  constexpr static auto kUserNumBytes = sizeof(Value);

  ~IntegerColumnReader() = default;

  // Opens the file.
  //
  // Args:
  //   path: Path to the file read read.
  //   max_value: Maximum possible represented value. Should be the same as the
  //     "max_value" used in the CategoricalColumnWriter.
  //   max_num_values: Maximum number of values returned in a single "Next"
  //     call.
  absl::Status Open(absl::string_view path, int64_t max_value,
                    int max_num_values);

  absl::Span<const Value> Values() override;

  absl::Status Next() override;

  absl::Status Close() override;

  // File data corresponding the last read values.
  absl::Span<const char> ActiveFileBuffer();

  // Number of bytes used to store one value in the file data.
  uint8_t file_num_bytes() const { return file_num_bytes_; }

 private:
  // File stream.
  file::FileInputByteStream file_;

  // Number of bytes used to store one value in the file data.
  uint8_t file_num_bytes_ = 0;

  // Buffer containing the last read values in the file format.
  std::vector<char> file_buffer_;

  // Buffer containing the last read values in the user format (i.e. "Value"
  // template). This buffer is only allocated / used if the file format is
  // different from the user format.
  std::vector<char> user_buffer_;

  // True iif. the file and user formats are the same.
  bool same_user_and_file_precision_ = false;

  // Last set of read values.
  absl::Span<const Value> values_;
};

// Reads a sequence of integer values from a sharded set of files.
// TODO(gbm): Multi-thread reading.
template <typename Value>
class ShardedIntegerColumnReader : public AbstractIntegerColumnIterator<Value> {
 public:
  ~ShardedIntegerColumnReader() {}

  // Opens the file.
  absl::Status Open(absl::string_view base_path, int64_t max_value,
                    int max_num_values, int begin_shard_idx, int end_shard_idx);

  absl::Span<const Value> Values() override;

  absl::Status Next() override;

  absl::Status Close() override;

  // Reads and appends the content of a sharded file.
  static absl::Status ReadAndAppend(absl::string_view base_path,
                                    int64_t max_value, int begin_shard_idx,
                                    int end_shard_idx,
                                    std::vector<Value>* output);

  // Part of the file buffer corresponding the current "Values";
  absl::Span<const char> ActiveFileBuffer() {
    return sub_reader_.ActiveFileBuffer();
  }

  uint8_t file_num_bytes() const { return sub_reader_.file_num_bytes(); }

 private:
  IntegerColumnReader<Value> sub_reader_;
  std::string base_path_;
  int64_t max_value_ = 0;
  int max_num_values_ = 0;
  int end_shard_idx_ = 0;
  int current_shard_idx_ = 0;
};

// Factory of integer column iterator that pre-loads/caches the sequence of
// integer in memory (in file format). Then the caller can call a
// "CreateIterator()" to get an iterator directly from memory. The values are
// decoded into the user format at iteration time (not during the initial file
// reading).
template <typename Value>
class InMemoryIntegerColumnReaderFactory {
 public:
  class InMemoryIntegerColumnReader
      : public AbstractIntegerColumnIterator<Value> {
   public:
    InMemoryIntegerColumnReader(
        const InMemoryIntegerColumnReaderFactory* const parent);

    InMemoryIntegerColumnReader(
        const InMemoryIntegerColumnReaderFactory* const parent,
        size_t begin_idx, size_t end_idx);

    ~InMemoryIntegerColumnReader() {}

    absl::Span<const Value> Values() override;

    absl::Status Next() override;

    absl::Status Close() override;

   private:
    // Buffer containing the last read values in the user format. Only used if
    // the file format is different from the user format.
    std::vector<char> user_buffer_;

    // Global index of the first value currently returned by "Values".
    size_t value_idx_ = 0;

    // End of the sequence of values to read.
    size_t end_idx_;

    // Current values returned by "Values".
    absl::Span<const Value> values_;

    const InMemoryIntegerColumnReaderFactory* const parent_ = nullptr;
  };

  // Number of bytes taken by one value stored in the user format.
  constexpr static auto kUserNumBytes = sizeof(Value);

  absl::Status Load(absl::string_view base_path, int64_t max_value,
                    int max_num_values, int begin_shard_idx, int end_shard_idx);

  void Reserve(size_t num_values, int64_t max_value);

  // Creates an iterator over the values. The factory owns the data and
  // should not be destroyed during the life of the iterator. Multiple iterators
  // can be created from the same factory and used concurrently.
  std::unique_ptr<InMemoryIntegerColumnReader> CreateIterator() const;
  std::unique_ptr<InMemoryIntegerColumnReader> CreateIterator(
      size_t begin_idx, size_t end_idx) const;

  // Amount of memory used by the container. Excludes the fixed size elements (a
  // few 100s of bytes).
  size_t MemoryUsage() { return file_buffer_.capacity() * sizeof(char); }

 private:
  // Values stored in file format.
  std::vector<char> file_buffer_;

  // Are the file and user formats the same?
  bool same_user_and_file_precision_ = false;

  // Maximum number of values returned in a single "Next" call.
  int max_num_values_ = 0;

  // Total number of values.
  size_t total_num_values_ = 0;

  // Number of bytes taken by a value in the file buffer.
  uint8_t file_num_bytes_ = 0;
};

// Writes a sequence of float values. Follows the same convention as
// "IntegerColumnWriter".
class FloatColumnWriter {
 public:
  absl::Status Open(absl::string_view path);
  absl::Status WriteValues(absl::Span<const float> values);
  absl::Status Close();

 private:
  file::FileOutputByteStream file_;

  // Open file.
  std::string path_;
};

// Returns a sequence of float values. Follows the same convention as
// "AbstractIntegerColumnIterator".
class AbstractFloatColumnIterator {
 public:
  virtual ~AbstractFloatColumnIterator() {}
  virtual absl::Span<const float> Values() = 0;
  virtual absl::Status Next() = 0;
  virtual absl::Status Close() = 0;
};

class FloatColumnReader : public AbstractFloatColumnIterator {
 public:
  ~FloatColumnReader() {}
  absl::Status Open(absl::string_view path, int max_num_values);
  absl::Span<const float> Values() override;
  absl::Status Next() override;
  absl::Status Close() override;

 private:
  file::FileInputByteStream file_;

  // Buffer containing the last read values.
  std::vector<float> buffer_;

  // Number of values read in the last "Next".
  int num_values_ = 0;
};

// Reads a sequence of float values from a sharded set of files. Follows the
// same convention as "ShardedIntegerColumnReader".
class ShardedFloatColumnReader : public AbstractFloatColumnIterator {
 public:
  ~ShardedFloatColumnReader() {}

  absl::Status Open(absl::string_view base_path, int max_num_values,
                    int begin_shard_idx, int end_shard_idx);
  absl::Span<const float> Values() override;
  absl::Status Next() override;
  absl::Status Close() override;

  // Reads and appends the content of a sharded file.
  static absl::Status ReadAndAppend(absl::string_view base_path,
                                    int begin_shard_idx, int end_shard_idx,
                                    std::vector<float>* output);

 private:
  FloatColumnReader sub_reader_;
  std::string base_path_;
  int max_num_values_ = 0;
  int end_shard_idx_ = 0;
  int current_shard_idx_ = 0;
};

// Loads a sequence of float in memory and make those values available through a
// "AbstractFloatColumnIterator". Follows the same convention as
// "InMemoryIntegerColumnReaderFactory".
class InMemoryFloatColumnReaderFactory {
 public:
  class InMemoryFloatColumnReader : public AbstractFloatColumnIterator {
   public:
    InMemoryFloatColumnReader(
        const InMemoryFloatColumnReaderFactory* const parent);

    ~InMemoryFloatColumnReader() = default;

    absl::Span<const float> Values() override;
    absl::Status Next() override;
    absl::Status Close() override;

   private:
    // Index of the first value currently returned by "Values".
    size_t value_idx_ = 0;

    // Current values.
    absl::Span<const float> values_;

    const InMemoryFloatColumnReaderFactory* const parent_ = nullptr;
  };

  void Reserve(size_t num_values);
  absl::Status Load(absl::string_view base_path, int max_num_values,
                    int begin_shard_idx, int end_shard_idx);

  std::unique_ptr<InMemoryFloatColumnReader> CreateIterator() const;
  size_t MemoryUsage() { return buffer_.capacity() * sizeof(float); }

 private:
  // All the values.
  std::vector<float> buffer_;

  // Maximum number of values returned in a single "Next" call.
  int max_num_values_ = 0;
};

// Indicates that a file will be written.
absl::Status PrepareOutputFile(absl::string_view path);

// Indicates that a file is done being written.
absl::Status FinalizeOutputFile(absl::string_view path);

}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_COLUMN_CACHE_H_
