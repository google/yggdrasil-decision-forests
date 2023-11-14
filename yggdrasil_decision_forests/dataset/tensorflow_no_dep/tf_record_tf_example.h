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

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_TENSORFLOW_NO_DEP_TF_RECORD_TF_EXAMPLE_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_TENSORFLOW_NO_DEP_TF_RECORD_TF_EXAMPLE_H_

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example_reader_interface.h"
#include "yggdrasil_decision_forests/dataset/example_writer_interface.h"
#include "yggdrasil_decision_forests/dataset/tensorflow/tf_example_io_interface.h"
#include "yggdrasil_decision_forests/dataset/tensorflow_no_dep/tf_record.h"

namespace yggdrasil_decision_forests::dataset::tensorflow_no_dep {

using TFRecordV2TFExampleReader = ShardedTFRecordReader<tensorflow::Example>;
REGISTER_AbstractTFExampleReader(TFRecordV2TFExampleReader,
                                 "FORMAT_TFE_TFRECORDV2");

class TFRecordV2TFEToExampleReaderInterface
    : public TFExampleReaderToExampleReader {
 public:
  TFRecordV2TFEToExampleReaderInterface(
      const proto::DataSpecification& data_spec,
      absl::optional<std::vector<int>> ensure_non_missing)
      : TFExampleReaderToExampleReader(data_spec, ensure_non_missing) {}

  std::unique_ptr<AbstractTFExampleReader> CreateReader() override {
    return absl::make_unique<TFRecordV2TFExampleReader>();
  }
};
REGISTER_ExampleReaderInterface(TFRecordV2TFEToExampleReaderInterface,
                                "FORMAT_TFE_TFRECORDV2");

class TFRecordV2TFExampleReaderToDataSpecCreator
    : public TFExampleReaderToDataSpecCreator {
  std::unique_ptr<AbstractTFExampleReader> CreateReader() override {
    return absl::make_unique<TFRecordV2TFExampleReader>();
  }
};

REGISTER_AbstractDataSpecCreator(TFRecordV2TFExampleReaderToDataSpecCreator,
                                 "FORMAT_TFE_TFRECORDV2");

// Write tf.Examples in TFRecords.
class TFRecordV2TFExampleWriter
    : public ShardedTFRecordWriter<tensorflow::Example> {};

REGISTER_AbstractTFExampleWriter(TFRecordV2TFExampleWriter,
                                 "FORMAT_TFE_TFRECORDV2");

class TFRecordV2TFEToExampleWriterInterface
    : public TFExampleWriterToExampleWriter {
 public:
  TFRecordV2TFEToExampleWriterInterface(
      const proto::DataSpecification& data_spec)
      : TFExampleWriterToExampleWriter(data_spec) {}

  std::unique_ptr<AbstractTFExampleWriter> CreateWriter() override {
    return absl::make_unique<TFRecordV2TFExampleWriter>();
  }
};
REGISTER_ExampleWriterInterface(TFRecordV2TFEToExampleWriterInterface,
                                "FORMAT_TFE_TFRECORDV2");

}  // namespace yggdrasil_decision_forests::dataset::tensorflow_no_dep

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_TENSORFLOW_NO_DEP_TF_RECORD_TF_EXAMPLE_H_
