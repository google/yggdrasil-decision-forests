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


#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_UTILS_STATUS_CASTERS_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_UTILS_STATUS_CASTERS_H_

#include <stdexcept>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

// C++ -> Python caster helpers.
//
// Failing statuses become Python exceptions; OK Status() becomes None.
//
// For code simplicity and to avoid clashes with dependencies, YDF uses a custom
// wrapper for absl::Status and absl::StatusOr<>.
//
// Usage example:
//
// m.def("my_func", WithStatus(MyFunc));
// m.def("my_func", WithStatusOr(MyFunc));
//
// Nonstatic member functions can be wrapped by passing a
// pointer-to-member-function:
// WithStatus(&MyClass::MyMethod)
// WithStatusOr(&MyClass::MyMethod)

inline void ThrowIfError(absl::Status src) {
  if (!src.ok()) {
    if (src.code() == absl::StatusCode::kInvalidArgument) {
      // Throws a ValueError in Python.
      throw std::invalid_argument(src.ToString());
    }
    throw std::runtime_error(src.ToString());
  }
}

// If one does not want to have to define a lambda specifying the inputs
// arguments, on can use the `WithStatus` / `WithStatusOr` wrappers.
//
// There are three specializations:
// - For free functions, `Sig` is the function type and `F` is `Sig&`.
// - For callable types, `Sig` is the pointer to member function type
//   and `F` is the type of the callable.
// - For a nonstatic member function of a class `C`, `Sig` is the function type
//   and `F` is Sig C::*.
//
// In the first two cases, the wrapper returns a callable with signature `Sig`;
// in the third case, the wrapper returns a callable with a modified signature
// that takes a C instance as the first argument.
template <typename Sig, typename F>
struct WithStatus;

// C++17 "deduction guide" that guides class template argument deduction (CTAD)
// For free functions.
template <typename F>
WithStatus(F) -> WithStatus<decltype(&F::operator()), F>;

// For callable types (with operator()).
template <typename... Args>
WithStatus(absl::Status (&)(Args...))
    -> WithStatus<absl::Status(Args...), absl::Status (&)(Args...)>;

// For unbound nonstatic member functions.
template <typename C, typename... Args>
WithStatus(absl::Status (C::*)(Args...))
    -> WithStatus<absl::Status(Args...), C>;

// Deduction guide for const methods.
template <typename C, typename... Args>
WithStatus(absl::Status (C::*)(Args...) const)
    -> WithStatus<absl::Status(Args...) const, C>;

// Template specializations.

// For free functions.
template <typename... Args>
struct WithStatus<absl::Status(Args...), absl::Status (&)(Args...)> {
  explicit WithStatus(absl::Status (&f)(Args...)) : func(f) {}
  void operator()(Args... args) {
    ThrowIfError(func(std::forward<Args>(args)...));
  }
  absl::Status (&func)(Args...);
};

// For callable types (with operator()), non-const and const versions.
template <typename C, typename... Args, typename F>
struct WithStatus<absl::Status (C::*)(Args...), F> {
  explicit WithStatus(F&& f) : func(std::move(f)) {}
  void operator()(Args... args) {
    ThrowIfError(func(std::forward<Args>(args)...));
  }
  F func;
};
template <typename C, typename... Args, typename F>
struct WithStatus<absl::Status (C::*)(Args...) const, F> {
  explicit WithStatus(F&& f) : func(std::move(f)) {}
  void operator()(Args... args) const {
    ThrowIfError(func(std::forward<Args>(args)...));
  }
  F func;
};

// For unbound nonstatic member functions, non-const and const versions.
// `ptmf` stands for "pointer to member function".
template <typename C, typename... Args>
struct WithStatus<absl::Status(Args...), C> {
  explicit WithStatus(absl::Status (C::*ptmf)(Args...)) : ptmf(ptmf) {}
  void operator()(C& instance, Args... args) {
    ThrowIfError((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::Status (C::*ptmf)(Args...);
};
template <typename C, typename... Args>
struct WithStatus<absl::Status(Args...) const, C> {
  explicit WithStatus(absl::Status (C::*ptmf)(Args...) const) : ptmf(ptmf) {}
  void operator()(const C& instance, Args... args) const {
    ThrowIfError((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::Status (C::*ptmf)(Args...) const;
};

// Utilities for `StatusOr`.

template <typename T>
T ValueOrThrow(absl::StatusOr<T> src) {
  if (!src.ok()) {
    if (src.status().code() == absl::StatusCode::kInvalidArgument) {
      // Throws a ValueError in Python.
      throw std::invalid_argument(src.status().ToString());
    }
    throw std::runtime_error(src.status().ToString());
  }
  return std::move(src).value();
}

template <typename Sig, typename F>
struct WithStatusOr;

template <typename F>
WithStatusOr(F) -> WithStatusOr<decltype(&F::operator()), F>;

template <typename R, typename... Args>
WithStatusOr(absl::StatusOr<R> (&)(Args...))
    -> WithStatusOr<absl::StatusOr<R>(Args...), absl::StatusOr<R> (&)(Args...)>;

template <typename C, typename R, typename... Args>
WithStatusOr(absl::StatusOr<R> (C::*)(Args...))
    -> WithStatusOr<absl::StatusOr<R>(Args...), C>;

// Deduction guide for const methods.
template <typename C, typename R, typename... Args>
WithStatusOr(absl::StatusOr<R> (C::*)(Args...) const)
    -> WithStatusOr<absl::StatusOr<R>(Args...) const, C>;

template <typename R, typename... Args>
struct WithStatusOr<absl::StatusOr<R>(Args...),
                    absl::StatusOr<R> (&)(Args...)> {
  explicit WithStatusOr(absl::StatusOr<R> (&f)(Args...)) : func(f) {}
  R operator()(Args... args) const {
    return ValueOrThrow(func(std::forward<Args>(args)...));
  }
  absl::StatusOr<R> (&func)(Args...);
};
template <typename R, typename C, typename... Args, typename F>
struct WithStatusOr<absl::StatusOr<R> (C::*)(Args...), F> {
  explicit WithStatusOr(F&& f) : func(std::move(f)) {}
  R operator()(Args... args) const {
    return ValueOrThrow(func(std::forward<Args>(args)...));
  }
  F func;
};
template <typename R, typename C, typename... Args, typename F>
struct WithStatusOr<absl::StatusOr<R> (C::*)(Args...) const, F> {
  explicit WithStatusOr(F&& f) : func(std::move(f)) {}
  R operator()(Args... args) const {
    return ValueOrThrow(func(std::forward<Args>(args)...));
  }
  F func;
};

// For unbound nonstatic member functions, non-const and const versions.
// `ptmf` stands for "pointer to member function".
template <typename R, typename C, typename... Args>
struct WithStatusOr<absl::StatusOr<R>(Args...), C> {
  explicit WithStatusOr(absl::StatusOr<R> (C::*ptmf)(Args...)) : ptmf(ptmf) {}
  R operator()(C& instance, Args... args) {
    return ValueOrThrow((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::StatusOr<R> (C::*ptmf)(Args...);
};
template <typename R, typename C, typename... Args>
struct WithStatusOr<absl::StatusOr<R>(Args...) const, C> {
  explicit WithStatusOr(absl::StatusOr<R> (C::*ptmf)(Args...) const)
      : ptmf(ptmf) {}
  R operator()(const C& instance, Args... args) const {
    return ValueOrThrow((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::StatusOr<R> (C::*ptmf)(Args...) const;
};

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_UTILS_STATUS_CASTERS_H_
