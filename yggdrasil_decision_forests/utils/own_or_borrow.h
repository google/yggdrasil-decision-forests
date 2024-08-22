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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_OWN_OR_BORROW_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_OWN_OR_BORROW_H_

#include <cstddef>
#include <vector>

#include "absl/types/span.h"

namespace yggdrasil_decision_forests::utils {

// Can own data similarly to a "vector<const T>" or borrow data similarly to a
// "Span<const T>".
template <typename T>
class VectorOwnOrBorrow {
 public:
  // Empty owned.
  VectorOwnOrBorrow() : values_(owned_values_), owner_(true) {}

  // Not copyable, not movable (for now).
  VectorOwnOrBorrow(const VectorOwnOrBorrow&) = delete;
  VectorOwnOrBorrow(VectorOwnOrBorrow&&) = delete;
  VectorOwnOrBorrow& operator=(const VectorOwnOrBorrow&) = delete;
  VectorOwnOrBorrow& operator=(VectorOwnOrBorrow&&) = delete;

  // Is the data owned.
  bool owner() const { return owner_; }

  // Accesses the data (owned or not owned).
  absl::Span<const T> values() const { return values_; }

  // Points to "src". The data is not owned. Release any previously owned data.
  void borrow(absl::Span<const T> src) {
    owner_ = false;
    owned_values_.clear();
    values_ = src;
  }

  void own(std::vector<T>&& src) {
    owner_ = true;
    owned_values_ = std::move(src);
    values_ = owned_values_;
  }

  size_t size() const { return values_.size(); }

 private:
  absl::Span<const T> values_;
  std::vector<T> owned_values_;
  bool owner_;
};

}  // namespace yggdrasil_decision_forests::utils

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_OWN_OR_BORROW_H_
