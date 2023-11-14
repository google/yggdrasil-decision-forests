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

// TODO: Remove YDF_INTERNAL_TF_EXAMPLE_H_INCLUDE.
#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_TF_EXAMPLE_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_TF_EXAMPLE_H_

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/tensorflow_no_dep/tf_example.h"
#include "yggdrasil_decision_forests/serving/example_set.h"

namespace yggdrasil_decision_forests {
namespace serving {

// Write the content of  TensorFlow Example into an ExampleSet.
absl::Status TfExampleToExampleSet(const tensorflow::Example& src,
                                   const int example_idx,
                                   const FeaturesDefinition& features,
                                   AbstractExampleSet* dst);

}  // namespace serving
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_TF_EXAMPLE_H_
