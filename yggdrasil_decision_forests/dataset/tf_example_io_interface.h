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

// This class contains utility functions to read and write tensorflow.Example
// from/to disk, as where as for the manipulation of tf.Examples.
//
// This file also contains wrappers for the simpleML Example reader (!=
// tf.Example) and dataspec creator from tf.Example readers.
//
// Usage example: Creating a reader of tensorflow.Examples contained in a
// RecordIO.
//
// auto reader =
// CreateTFExampleReaderCreateTFExampleReader("recordio+tfe:/my/file@10");
//
// The format of the file is defined with a prefix. The available prefixes are
// listed in "yggdrasil_decision_forests/dataset/formats.h".
//
// In addition to this lib (i.e. :tf_example_io_interface), the user should add
// dependency to the ":tf_example_io_*" libraries corresponding to the desired
// supported formats. For example, the dependency to ":tf_example_io_recordio"
// add support for tensorflow.Example stored in recordIOs.
//
// Alternatively, a dependency to ":all_tf_example_io_interfaces" can be created
// to add support to all supported formats.

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_TF_EXAMPLE_IO_INTERFACE_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_TF_EXAMPLE_IO_INTERFACE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/example_reader_interface.h"
#include "yggdrasil_decision_forests/dataset/example_writer_interface.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/registration.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Abstract class to read a stream of tf.Examples from a sharded dataset (e.g.
// a sharded sstable).
using AbstractTFExampleReader = utils::ShardedReader<tensorflow::Example>;

REGISTRATION_CREATE_POOL(AbstractTFExampleReader);

#define REGISTER_AbstractTFExampleReader(name, key) \
  REGISTRATION_REGISTER_CLASS(name, key, AbstractTFExampleReader)

// Abstract class to write a stream of tf.Examples to a sharded dataset (e.g.
// a sharded sstable).
using AbstractTFExampleWriter = utils::ShardedWriter<tensorflow::Example>;

REGISTRATION_CREATE_POOL(AbstractTFExampleWriter);

#define REGISTER_AbstractTFExampleWriter(name, key) \
  REGISTRATION_REGISTER_CLASS(name, key, AbstractTFExampleWriter)

// Creates a tf.example read from a type sharded path.
utils::StatusOr<std::unique_ptr<AbstractTFExampleReader>> CreateTFExampleReader(
    absl::string_view typed_path);

// Creates a tf.example writer from a type sharded path.
utils::StatusOr<std::unique_ptr<AbstractTFExampleWriter>> CreateTFExampleWriter(
    absl::string_view typed_path, int64_t num_records_by_shard);

// Wrapper around a TFExampleReader with the signature of an Example reader.
// This class is thread-compatible (you can use it from multiple threads at the
// same time, but you are in charge of the synchronization).
class TFExampleReaderToExampleReader : public ExampleReaderInterface {
 public:
  virtual std::unique_ptr<AbstractTFExampleReader> CreateReader() = 0;

  TFExampleReaderToExampleReader(
      const proto::DataSpecification& data_spec,
      absl::optional<std::vector<int>> ensure_non_missing);

  absl::Status Open(absl::string_view sharded_path) override;

  utils::StatusOr<bool> Next(proto::Example* example) override;

 private:
  std::unique_ptr<AbstractTFExampleReader> tf_reader_;
  tensorflow::Example tfexample_buffer_;
  const proto::DataSpecification data_spec_;
  const absl::optional<std::vector<int>> ensure_non_missing_;
};

// Wrapper around a TFExampleReader with the signature of an dataspec generator.
class TFExampleReaderToDataSpecCreator : public AbstractDataSpecCreator {
 public:
  virtual std::unique_ptr<AbstractTFExampleReader> CreateReader() = 0;

  void InferColumnsAndTypes(const std::vector<std::string>& paths,
                            const proto::DataSpecificationGuide& guide,
                            proto::DataSpecification* data_spec) override;

  void ComputeColumnStatistics(
      const std::vector<std::string>& paths,
      const proto::DataSpecificationGuide& guide,
      proto::DataSpecification* data_spec,
      proto::DataSpecificationAccumulator* accumulator) override;

  utils::StatusOr<int64_t> CountExamples(absl::string_view path) override;
};

// Example writer made as a wrapper around a tf.Example writer.
// This class is thread-compatible (you can use it from multiple threads at the
// same time, but you are in charge of the synchronization).
class TFExampleWriterToExampleWriter : public ExampleWriterInterface {
 public:
  virtual std::unique_ptr<AbstractTFExampleWriter> CreateWriter() = 0;

  TFExampleWriterToExampleWriter(const proto::DataSpecification& data_spec);

  absl::Status Open(absl::string_view sharded_path,
                    int64_t num_records_by_shard) override;

  absl::Status Write(const proto::Example& example) override;

 private:
  std::unique_ptr<AbstractTFExampleWriter> tf_writer_;
  const proto::DataSpecification data_spec_;
  tensorflow::Example tfexample_buffer_;
};

// Determine the most likely type of the attribute according to the current
// most likely value type and an observed string value.
proto::ColumnType InferType(const proto::DataSpecificationGuide& guide,
                            const tensorflow::Feature& feature,
                            const proto::Tokenizer& tokenizer,
                            proto::ColumnType previous_type,
                            int* num_sub_values);

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_TF_EXAMPLE_IO_INTERFACE_H_
