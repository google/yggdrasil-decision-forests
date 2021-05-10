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

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_WRITER_INTERFACE_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_WRITER_INTERFACE_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/registration.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Interface to write a stream of proto::Example.
class ExampleWriterInterface {
 public:
  virtual ~ExampleWriterInterface() = default;

  virtual absl::Status Open(absl::string_view sharded_path,
                            int64_t num_records_by_shard) = 0;

  // Write an example.
  virtual absl::Status Write(const proto::Example& example) = 0;
};

REGISTRATION_CREATE_POOL(ExampleWriterInterface,
                         const proto::DataSpecification&);

#define REGISTER_ExampleWriterInterface(name, key) \
  REGISTRATION_REGISTER_CLASS(name, key, ExampleWriterInterface)

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_WRITER_INTERFACE_H_
