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

// Base interface for a stream of example.
//
#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_READER_INTERFACE_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_READER_INTERFACE_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/registration.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Interface to read a stream of proto::Example.
class ExampleReaderInterface {
 public:
  virtual ~ExampleReaderInterface() = default;

  virtual absl::Status Open(absl::string_view sharded_path) = 0;

  // Tries to retrieve the next available example. If no more examples are
  // available, returns false.
  virtual utils::StatusOr<bool> Next(proto::Example* example) = 0;
};

REGISTRATION_CREATE_POOL(ExampleReaderInterface,
                         const proto::DataSpecification&,
                         absl::optional<std::vector<int>>);

#define REGISTER_ExampleReaderInterface(name, key) \
  REGISTRATION_REGISTER_CLASS(name, key, ExampleReaderInterface)

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_READER_INTERFACE_H_
