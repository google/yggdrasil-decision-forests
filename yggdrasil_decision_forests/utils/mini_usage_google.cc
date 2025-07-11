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

#include "yggdrasil_decision_forests/utils/mini_usage_google.h"

#ifdef YGG_ENABLE_MINI_USAGE
#include <cstdint>
#include <string>
#include <string_view>

#include "monitoring/streamz/public/concurrent.h"
#include "monitoring/streamz/public/metadata.h"
#include "absl/strings/str_cat.h"

#endif

namespace yggdrasil_decision_forests::utils::usage {

#ifdef YGG_ENABLE_MINI_USAGE
namespace {

const char kMetricNumInference[] =
    "/learning/simple_ml/inference/num_inferences";

// Fields
const char kFieldOwner[] = "owner";
const char kFieldCreated_date[] = "created_date";
const char kFieldUid[] = "uid";
const char kFieldFramework[] = "framework";

using ::monitoring::streamz::CpuLocalCounter;
using ::monitoring::streamz::Metadata;

// Note: Monarch does not support (u)int64 fields. We have to cast them to
// string: https://yaqs.corp.google.com/eng/q/6046095177285632
auto* const inference_counter =
    CpuLocalCounter<std::string, std::string, std::string, std::string>::New(
        kMetricNumInference, kFieldOwner, kFieldCreated_date, kFieldUid,
        kFieldFramework,
        Metadata("Number of inference calls to a simpleML model."));

}  // namespace
#endif

void OnInference(const OnInferenceArg& arg) {
#ifdef YGG_ENABLE_MINI_USAGE
  inference_counter->IncrementBy(arg.num_examples, arg.owner,
                                 absl::StrCat(arg.created_date),
                                 absl::StrCat(arg.uid), arg.framework);
#endif
}

}  // namespace yggdrasil_decision_forests::utils::usage
