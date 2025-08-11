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

#include "yggdrasil_decision_forests/model/metadata.h"

#include <string>

#include "absl/log/log.h"

namespace yggdrasil_decision_forests {
namespace model {

void MetaData::Export(model::proto::Metadata* dst) const {
  dst->set_owner(owner_);
  dst->set_created_date(created_date_);
  dst->set_uid(uid_);
  dst->set_framework(framework_);
  dst->mutable_custom_fields()->Clear();
  for (const auto& field : custom_fields_) {
    auto* proto_field = dst->mutable_custom_fields()->Add();
    proto_field->set_key(field.first);
    proto_field->set_value(field.second);
  }
}

void MetaData::Import(const model::proto::Metadata& src) {
  owner_ = src.owner();
  created_date_ = src.created_date();
  uid_ = src.uid();
  framework_ = src.framework();
  custom_fields_.clear();
  custom_fields_.reserve(src.custom_fields_size());
  for (const auto& field : src.custom_fields()) {
    std::string key = field.key();
    if (custom_fields_.contains(key)) {
      LOG(WARNING) << "This model contains duplicate key " << key
                   << " in the custom field's of the model metadata. This is "
                      "not supported. The custom fields of the model metadata "
                      "may not be fully loaded.";
    }
    custom_fields_[key] = field.value();
  }
}

}  // namespace model
}  // namespace yggdrasil_decision_forests
