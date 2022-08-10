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

#include "yggdrasil_decision_forests/dataset/tensorflow/tf_example.h"

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"

namespace yggdrasil_decision_forests {
namespace dataset {

absl::Status TfExampleToYdfExample(const ::tensorflow::Example& tf_example,
                                   const proto::DataSpecification& data_spec,
                                   proto::Example* example) {
  return TfExampleToExample(tf_example, data_spec, example);
}

absl::Status YdfExampleToTfExample(const proto::Example& example,
                                   const proto::DataSpecification& data_spec,
                                   ::tensorflow::Example* tf_example) {
  return ExampleToTfExampleWithStatus(example, data_spec, tf_example);
}

absl::Status TfExampleToExampleSet(const ::tensorflow::Example& src,
                                   int example_idx,
                                   const serving::FeaturesDefinition& features,
                                   serving::AbstractExampleSet* dst) {
  return dst->FromTensorflowExample(src, example_idx, features);
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
