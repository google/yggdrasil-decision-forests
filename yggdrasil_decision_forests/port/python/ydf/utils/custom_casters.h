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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_UTILS_CUSTOM_CASTERS_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_UTILS_CUSTOM_CASTERS_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "absl/container/flat_hash_map.h"

namespace pybind11 {
namespace detail {
// Convert between absl::flat_hash_map and python dict just like std::map.
template <typename Key, typename Value, typename Hash, typename Equal,
          typename Alloc>
struct type_caster<absl::flat_hash_map<Key, Value, Hash, Equal, Alloc>>
    : map_caster<absl::flat_hash_map<Key, Value, Hash, Equal, Alloc>, Key,
                 Value> {};
}  // namespace detail
}  // namespace pybind11

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_UTILS_CUSTOM_CASTERS_H_
