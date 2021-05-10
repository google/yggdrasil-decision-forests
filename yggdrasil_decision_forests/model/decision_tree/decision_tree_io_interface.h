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

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_IO_INTERFACE_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_IO_INTERFACE_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/registration.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// Instantiate of format IO implementation.
class AbstractFormat;

// Creates an "AbstractFormat" corresponding to the "format". If not
// implementation is registered with this format, returns an error.
//
// The returned "AbstractFormat" can then be used to create a sequential reader
// or writer of nodes.
utils::StatusOr<std::unique_ptr<AbstractFormat>> GetFormatImplementation(
    absl::string_view format);

// Containers/formats available for save/load decision trees to/from disk.
class AbstractFormat {
 public:
  virtual ~AbstractFormat() = default;
  virtual std::unique_ptr<utils::ShardedReader<proto::Node>> CreateReader()
      const = 0;
  virtual std::unique_ptr<utils::ShardedWriter<proto::Node>> CreateWriter()
      const = 0;
};

REGISTRATION_CREATE_POOL(AbstractFormat);

#define REGISTER_AbstractFormat(name, key) \
  REGISTRATION_REGISTER_CLASS(name, key, AbstractFormat)

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_IO_INTERFACE_H_
