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

// This file contains copy of method that exist in the library we are using,
// but not in the version we are using.
//
// For each method, write what is the source library, and when it will be
// possible to use the source library.
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_COMPATIBILITY_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_COMPATIBILITY_H_

#include <stdint.h>

#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/types/optional.h"

#if defined YGG_ABSL_NO_STATUSOR
#include "absl/status/status.h"
#else
#include "absl/status/statusor.h"
#endif

namespace yggdrasil_decision_forests {
namespace utils {

// Name of the user.
inline absl::optional<std::string> UserName() {
   // TODO(gbm): Platform specific implementation.
   return {};
}

#if defined YGG_ABSL_NO_STATUSOR

// Minimal version of absl::StatusOr for Yggdrasil's needs.
//
// TensorFlow cannot be compiled with two different versions of Absl.
// The version of Absl currently used by OSS TensorFlow does not have
// absl::StatusOr. Remove this implementation of StatusOr once TensorFlow uses a
// more recent version of Absl.
template <typename T>
class StatusOr {
 private:
  inline void CheckOk(const absl::Status& status) {
    if (ABSL_PREDICT_FALSE(!status.ok())) {
      std::cerr << "Attempting to fetch value instead of handling error "
                << status.ToString();
      std::exit(1);
    }
  }

  inline void CheckNoOk(const absl::Status& status) {
    if (ABSL_PREDICT_FALSE(status.ok())) {
      std::cerr << "Cannot set a StatusOr with OK status";
      std::exit(1);
    }
  }

 public:
  typedef T value_type;

  explicit StatusOr() {}

  // Status constructors / assignation.
  StatusOr(const absl::Status& status) : status_(status) { CheckNoOk(status_); }

  StatusOr(absl::Status&& status) : status_(std::move(status)) {
    CheckNoOk(status_);
  }

  // StatusOr constructors / assignation.

  StatusOr(const StatusOr& other) : status_(other.status_) {
    new (&data_) T(other.data_);
  }

  StatusOr(StatusOr&& other) : status_(std::move(other.status_)) {
    new (&data_) T(std::move(other.data_));
  }

  StatusOr& operator=(const StatusOr<T>& other) {
    if (status_.ok() && !other.status_.ok()) {
      data_.~T();
    } else if (!status_.ok() && other.status_.ok()) {
      new (&data_) T(other.data_);
    } else if (status_.ok() && other.status_.ok()) {
      data_ = other.data_;
    }
    status_ = other.status_;
    return *this;
  }

  StatusOr& operator=(StatusOr<T>&& other) {
    if (status_.ok() && !other.status_.ok()) {
      data_.~T();
    } else if (!status_.ok() && other.status_.ok()) {
      new (&data_) T(std::move(other.data_));
    } else if (status_.ok() && other.status_.ok()) {
      data_ = std::move(other.data_);
    }
    status_ = other.status_;
    return *this;
  }

  // Data constructors / assignation.
  template <typename U>
  StatusOr(const U& data) : status_(absl::OkStatus()) {
    new (&data_) T(data);
  }

  template <typename U>
  StatusOr(U&& data) : status_(absl::OkStatus()) {
    new (&data_) T(std::move(data));
  }

  template <typename U>
  StatusOr& operator=(const U& other) {
    if (status_.ok()) {
      data_ = other;
    } else {
      new (&data_) T(other);
    }
    status_ = absl::OkStatus();
    return *this;
  }

  template <typename U>
  StatusOr& operator=(U&& other) {
    if (status_.ok()) {
      data_ = std::move(other);
    } else {
      new (&data_) T(std::move(other));
    }
    status_ = absl::OkStatus();
    return *this;
  }

  // Destructor.
  ~StatusOr() {
    if (status_.ok()) {
      data_.~T();
    }
  }

  // Accessors.
  ABSL_MUST_USE_RESULT bool ok() const { return status_.ok(); }

  ABSL_MUST_USE_RESULT const absl::Status& status() const& { return status_; }

  const T& value() const& { return data_; }

  T& value() & { return data_; }

  T&& value() && { return std::move(data_); }

  const T& ValueOrDie() const& {
    CheckOk(status_);
    return data_;
  }

  T& ValueOrDie() & {
    CheckOk(status_);
    return data_;
  }

  T&& ValueOrDie() && {
    CheckOk(status_);
    return std::move(data_);
  }

 private:
  absl::Status status_{absl::InternalError("")};

  union {
    // "data_" is initialized iif. "status_.ok()" is true.
    T data_;
  };
};

#else
template <typename T>
using StatusOr = ::absl::StatusOr<T>;
#endif

// Same as std::clamp in >=c++17.
template <class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
  assert(!(hi < lo));
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

#if defined(__GNUC__)
#define PREFETCH(addr) __builtin_prefetch(addr)
#else
#define PREFETCH(addr)
#endif

// Same as std::is_same_v in >=c++17.
template <class T, class U>
constexpr bool is_same_v = ::std::is_same<T, U>::value;

// Same as ABSL_INTERNAL_ASSUME(n != 0) + absl::countr_zero.
// Remove when Absl release a new TLS version, and when TensorFlow supports it.
//
// This code was copied from: absl/numeric/internal/bits.h
ABSL_ATTRIBUTE_ALWAYS_INLINE inline int CountTrailingZeroesNonzero64(
    uint64_t x) {
#if ABSL_HAVE_BUILTIN(__builtin_ctzll)
  static_assert(sizeof(unsigned long long) == sizeof(x),  // NOLINT(runtime/int)
                "__builtin_ctzll does not take 64-bit arg");
  return __builtin_ctzll(x);
#elif defined(_MSC_VER) && !defined(__clang__) && \
    (defined(_M_X64) || defined(_M_ARM64))
  unsigned long result = 0;  // NOLINT(runtime/int)
  _BitScanForward64(&result, x);
  return result;
#elif defined(_MSC_VER) && !defined(__clang__)
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (static_cast<uint32_t>(x) == 0) {
    _BitScanForward(&result, static_cast<unsigned long>(x >> 32));
    return result + 32;
  }
  _BitScanForward(&result, static_cast<unsigned long>(x));
  return result;
#else
  int c = 63;
  x &= ~x + 1;
  if (x & 0x00000000FFFFFFFF) c -= 32;
  if (x & 0x0000FFFF0000FFFF) c -= 16;
  if (x & 0x00FF00FF00FF00FF) c -= 8;
  if (x & 0x0F0F0F0F0F0F0F0F) c -= 4;
  if (x & 0x3333333333333333) c -= 2;
  if (x & 0x5555555555555555) c -= 1;
  return c;
#endif
}

// Same as absl's CountLeadingZeroes64.
// Remove when Absl release a new TLS version, and when TensorFlow supports it.
//
// This code was copied from: absl/numeric/internal/bits.h
ABSL_ATTRIBUTE_ALWAYS_INLINE
inline int CountLeadingZeroes64(uint64_t x) {
#if ABSL_NUMERIC_INTERNAL_HAVE_BUILTIN_OR_GCC(__builtin_clzll)
  // Use __builtin_clzll, which uses the following instructions:
  //  x86: bsr, lzcnt
  //  ARM64: clz
  //  PPC: cntlzd
  static_assert(sizeof(unsigned long long) == sizeof(x),  // NOLINT(runtime/int)
                "__builtin_clzll does not take 64-bit arg");

  // Handle 0 as a special case because __builtin_clzll(0) is undefined.
  return x == 0 ? 64 : __builtin_clzll(x);
#elif defined(_MSC_VER) && !defined(__clang__) && \
    (defined(_M_X64) || defined(_M_ARM64))
  // MSVC does not have __buitin_clzll. Use _BitScanReverse64.
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (_BitScanReverse64(&result, x)) {
    return 63 - result;
  }
  return 64;
#elif defined(_MSC_VER) && !defined(__clang__)
  // MSVC does not have __buitin_clzll. Compose two calls to _BitScanReverse
  unsigned long result = 0;  // NOLINT(runtime/int)
  if ((x >> 32) &&
      _BitScanReverse(&result, static_cast<unsigned long>(x >> 32))) {
    return 31 - result;
  }
  if (_BitScanReverse(&result, static_cast<unsigned long>(x))) {
    return 63 - result;
  }
  return 64;
#else
  int zeroes = 60;
  if (x >> 32) {
    zeroes -= 32;
    x >>= 32;
  }
  if (x >> 16) {
    zeroes -= 16;
    x >>= 16;
  }
  if (x >> 8) {
    zeroes -= 8;
    x >>= 8;
  }
  if (x >> 4) {
    zeroes -= 4;
    x >>= 4;
  }
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[x] + zeroes;
#endif
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_COMPAT_H_
