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

// Support for dataset stored as TFRecord of tf.Examples.
//
#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_TF_EXAMPLE_IO_TFRECORD_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_TF_EXAMPLE_IO_TFRECORD_H_

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/tstring.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example_reader_interface.h"
#include "yggdrasil_decision_forests/dataset/example_writer_interface.h"
#include "yggdrasil_decision_forests/dataset/tf_example_io_interface.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/sharded_io_tfrecord.h"

namespace yggdrasil_decision_forests {
namespace dataset {

using TFRecordTFExampleReader =
    utils::TFRecordShardedReader<tensorflow::Example>;
REGISTER_AbstractTFExampleReader(TFRecordTFExampleReader,
                                 "FORMAT_TFE_TFRECORD");

class TFRecordTFEToExampleReaderInterface
    : public TFExampleReaderToExampleReader {
 public:
  TFRecordTFEToExampleReaderInterface(
      const proto::DataSpecification& data_spec,
      absl::optional<std::vector<int>> ensure_non_missing)
      : TFExampleReaderToExampleReader(data_spec, ensure_non_missing) {}

  std::unique_ptr<AbstractTFExampleReader> CreateReader() override {
    return absl::make_unique<TFRecordTFExampleReader>();
  }
};
REGISTER_ExampleReaderInterface(TFRecordTFEToExampleReaderInterface,
                                "FORMAT_TFE_TFRECORD");

class TFRTFExampleReaderToDataSpecCreator
    : public TFExampleReaderToDataSpecCreator {
  std::unique_ptr<AbstractTFExampleReader> CreateReader() override {
    return absl::make_unique<TFRecordTFExampleReader>();
  }
};

REGISTER_AbstractDataSpecCreator(TFRTFExampleReaderToDataSpecCreator,
                                 "FORMAT_TFE_TFRECORD");

// Write tf.Examples in TFRecords.
class TFRecordTFExampleWriter
    : public utils::TFRecordShardedWriter<tensorflow::Example> {};

REGISTER_AbstractTFExampleWriter(TFRecordTFExampleWriter,
                                 "FORMAT_TFE_TFRECORD");

class TFRecordTFEToExampleWriterInterface
    : public TFExampleWriterToExampleWriter {
 public:
  TFRecordTFEToExampleWriterInterface(const proto::DataSpecification& data_spec)
      : TFExampleWriterToExampleWriter(data_spec) {}

  std::unique_ptr<AbstractTFExampleWriter> CreateWriter() override {
    return absl::make_unique<TFRecordTFExampleWriter>();
  }
};
REGISTER_ExampleWriterInterface(TFRecordTFEToExampleWriterInterface,
                                "FORMAT_TFE_TFRECORD");

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_TF_EXAMPLE_IO_TFRECORD_H_
