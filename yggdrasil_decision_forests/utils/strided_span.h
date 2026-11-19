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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_STRIDED_SPAN_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_STRIDED_SPAN_H_

#include <cstddef>
#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/types/span.h"

namespace yggdrasil_decision_forests::utils {

// A read-only, non-owning view over evenly spaced values. Similar to a
// "absl::Span<const T>", but consecutive values can be separated by a constant
// gap.
//
// "StridedSpan" does not own the underlying data. The data should outlive the
// "StridedSpan".
template <typename T>
class ABSL_ATTRIBUTE_VIEW StridedSpan {
 public:
  using value_type = std::remove_cv_t<T>;

  constexpr StridedSpan() noexcept = default;

  // "size" values starting at "data" and separated by "item_stride" items, i.e.
  // the i-th value is "data[i * item_stride]". "item_stride" can be negative or
  // zero.
  constexpr StridedSpan(const T* data ABSL_ATTRIBUTE_LIFETIME_BOUND,
                        const size_t size, const ptrdiff_t item_stride) noexcept
      : values_(data), size_(size), item_stride_(item_stride) {}

  // Contiguous values.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr StridedSpan(absl::Span<const T> values) noexcept
      : StridedSpan(values.data(), values.size(), /*item_stride=*/1) {}

  template <typename LazyT = T,
            typename = std::enable_if_t<!std::is_const_v<LazyT>>>
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr StridedSpan(absl::Span<LazyT> values) noexcept
      : StridedSpan(values.data(), values.size(), /*item_stride=*/1) {}

  // Contiguous container (e.g. std::vector<T>, std::array<T, N>), avoid
  // nesting.
  template <typename C,
            typename RawC = std::remove_cv_t<std::remove_reference_t<C>>,
            typename = std::enable_if_t<
                !std::is_same_v<RawC, StridedSpan> &&
                !std::is_same_v<RawC, absl::Span<T>> &&
                !std::is_same_v<RawC, absl::Span<const T>> &&
                std::is_convertible_v<const C&, absl::Span<const T>>>>
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr StridedSpan(const C& values ABSL_ATTRIBUTE_LIFETIME_BOUND) noexcept
      : StridedSpan(absl::Span<const T>(values)) {}

  // Number of values.
  constexpr size_t size() const noexcept { return size_; }

  constexpr bool empty() const noexcept { return size_ == 0; }

  // Distance, in number of items, between two consecutive values.
  constexpr ptrdiff_t item_stride() const noexcept { return item_stride_; }

  // Pointer to the first value.
  constexpr const T* data() const noexcept { return values_; }

  // Accesses a value. "index" should be in [0, size).
  const T& operator[](const size_t index) const {
    ABSL_HARDENING_ASSERT(index < size_);
    return values_[static_cast<ptrdiff_t>(index) * item_stride_];
  }

  // Sub-span of "count" values starting at "begin".
  StridedSpan subspan(const size_t begin, const size_t count) const {
    ABSL_HARDENING_ASSERT(begin <= size_);
    ABSL_HARDENING_ASSERT(count <= size_ - begin);
    if (count == 0) {
      return StridedSpan(nullptr, 0, item_stride_);
    }
    return StridedSpan(values_ + static_cast<ptrdiff_t>(begin) * item_stride_,
                       count, item_stride_);
  }

 private:
  const T* values_ = nullptr;
  size_t size_ = 0;
  ptrdiff_t item_stride_ = 1;
};

}  // namespace yggdrasil_decision_forests::utils

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_STRIDED_SPAN_H_
