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

#include <memory>

#include "yggdrasil_decision_forests/utils/sharded_io_tfrecord.h"

#include "absl/memory/memory.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree_io_interface.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// TFRecord container.
class TFRecordFormat : public AbstractFormat {
 public:
  ~TFRecordFormat() override = default;

  std::unique_ptr<utils::ShardedReader<proto::Node>> CreateReader()
      const override {
    return absl::make_unique<utils::TFRecordShardedReader<proto::Node>>();
  };

  std::unique_ptr<utils::ShardedWriter<proto::Node>> CreateWriter()
      const override {
    return absl::make_unique<utils::TFRecordShardedWriter<proto::Node>>();
  };
};
REGISTER_AbstractFormat(TFRecordFormat, "TFE_TFRECORD");

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests