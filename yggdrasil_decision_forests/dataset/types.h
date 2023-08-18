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

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_TYPES_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_TYPES_H_

#include <cstdint>

#include "absl/status/status.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// "ExampleIdx" is a signed integer able to store the number of examples in a
// training dataset.
//
// ExampleIdx is controlled by the --define=ydf_example_idx_num_bits={32,64}
// flag. See the documentation of this flag for more details.
#if defined(YGGDRASIL_EXAMPLE_IDX_32_BITS)
typedef int32_t SignedExampleIdx;
typedef uint32_t UnsignedExampleIdx;
#elif defined(YGGDRASIL_EXAMPLE_IDX_64_BITS)
typedef int64_t SignedExampleIdx;
typedef uint64_t UnsignedExampleIdx;
#else
#warning "Neither YGGDRASIL_EXAMPLE_IDX_{32,64}_BITS is defined"
#endif

// Checks at runtime that the number of examples is compatible with
// "SignedExampleIdx".
absl::Status CheckNumExamples(size_t num_examples);

}  // namespace dataset

namespace model {
// Alias in "model" namespace.
typedef dataset::SignedExampleIdx SignedExampleIdx;
typedef dataset::UnsignedExampleIdx UnsignedExampleIdx;
}  // namespace model

}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_TYPES_H_
