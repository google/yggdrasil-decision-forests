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

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_TENSORFLOW_NO_DEP_TF_EXAMPLE_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_TENSORFLOW_NO_DEP_TF_EXAMPLE_H_

#include "absl/strings/string_view.h"

#ifdef YGG_USE_YDF_TENSORFLOW_PROTO
#include "yggdrasil_decision_forests/dataset/tensorflow_no_dep/example.pb.h"
#else
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#endif

namespace yggdrasil_decision_forests::tensorflow {
#ifdef YGG_USE_YDF_TENSORFLOW_PROTO
typedef ::yggdrasil_decision_forests::tensorflow_no_dep::Example Example;
typedef ::yggdrasil_decision_forests::tensorflow_no_dep::Feature Feature;
#else
typedef ::tensorflow::Example Example;
typedef ::tensorflow::Feature Feature;
#endif

void SetFeatureValues(std::initializer_list<double> values,
                      absl::string_view key, tensorflow::Example* example);
void SetFeatureValues(std::initializer_list<int> values, absl::string_view key,
                      tensorflow::Example* example);
void SetFeatureValues(std::initializer_list<absl::string_view> values,
                      absl::string_view key, tensorflow::Example* example);

}  // namespace yggdrasil_decision_forests::tensorflow

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_TENSORFLOW_NO_DEP_TF_EXAMPLE_H_
