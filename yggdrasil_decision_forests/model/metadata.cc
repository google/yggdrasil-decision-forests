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

#include "yggdrasil_decision_forests/model/metadata.h"

namespace yggdrasil_decision_forests {
namespace model {

void MetaData::Export(model::proto::Metadata* dst) const {
  dst->set_owner(owner_);
  dst->set_created_date(created_date_);
  dst->set_uid(uid_);
  dst->set_framework(framework_);
}

void MetaData::Import(const model::proto::Metadata& src) {
  owner_ = src.owner();
  created_date_ = src.created_date();
  uid_ = src.uid();
  framework_ = src.framework();
}

}  // namespace model
}  // namespace yggdrasil_decision_forests
