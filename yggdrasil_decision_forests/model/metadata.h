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

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_METADATA_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_METADATA_H_

#include <string>

#include "yggdrasil_decision_forests/model/abstract_model.pb.h"

namespace yggdrasil_decision_forests {
namespace model {

// Meta-data about the model.
class MetaData {
 public:
  const std::string& owner() const { return owner_; }
  void set_owner(const std::string& value) { owner_ = value; }

  const uint64_t created_date() const { return created_date_; }
  void set_created_date(const uint64_t value) { created_date_ = value; }

  const uint64_t uid() const { return uid_; }
  void set_uid(const uint64_t value) { uid_ = value; }

  const std::string& framework() const { return framework_; }
  void set_framework(const std::string& value) { framework_ = value; }

  void Export(model::proto::Metadata* dst) const;
  void Import(const model::proto::Metadata& src);

 private:
  std::string owner_;
  int64_t created_date_ = 0;
  uint64_t uid_ = 0;
  std::string framework_;
};

}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_METADATA_H_
