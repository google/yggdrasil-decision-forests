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

// Registration of abstract classes by a string name.
// All the user operations (i.e. outside of "internal::") are thread safe.
//
// Usage example:
//
// interface.h
//   #include "registration.h"
//   class BaseClass {};
//   REGISTRATION_CREATE_POOL(BaseClass, absl::string_view);
//
// implementation1.cc
//   #include "interface.h"
//   class Implementation1 : public BaseClass {
//     Implementation1(absl::string_view name) {...}
//   REGISTRATION_REGISTER_CLASS(Implementation1, "Implementation1", BaseClass);
//
// consumer.cc
//   #include "interface.h"
//   BaseClassRegisterer::Create("C1", "Toto").value() ...
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_REGISTRATION_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_REGISTRATION_H_

#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

// Build the registration pool name from the interface class name.
#define INTERNAL_REGISTERER_CLASSNAME(INTERFACE) INTERFACE##Registerer

// Register a new pool of classes.
//
// Args:
//   cls: Interface class.
//   ...: Argument of the implementation class constructors.
//
#define REGISTRATION_CREATE_POOL(INTERFACE, ...)                             \
  class INTERNAL_REGISTERER_CLASSNAME(INTERFACE)                             \
      : public ::yggdrasil_decision_forests::registration::internal::        \
            ClassPool<INTERFACE, ##__VA_ARGS__> {                            \
   public:                                                                   \
    template <typename IMPLEMENTATION>                                       \
    static ::yggdrasil_decision_forests::registration::internal::Empty       \
    Register(const absl::string_view key) {                                  \
      if (IsName(key)) return {};                                            \
      utils::concurrency::MutexLock l(                                       \
          &::yggdrasil_decision_forests::registration::internal::            \
              registration_mutex);                                           \
      InternalGetItems()->push_back(                                         \
          absl::make_unique<                                                 \
              ::yggdrasil_decision_forests::registration::internal::Creator< \
                  INTERFACE, IMPLEMENTATION, ##__VA_ARGS__>>(key));          \
      return {};                                                             \
    }                                                                        \
  };

// Adds an implementation class to an existing pool.
#define REGISTRATION_REGISTER_CLASS(IMPLEMENTATION, name, INTERFACE)      \
  static const auto register_##IMPLEMENTATION##_in_##INTERFACE =          \
      INTERNAL_REGISTERER_CLASSNAME(INTERFACE)::Register<IMPLEMENTATION>( \
          name);

namespace yggdrasil_decision_forests {
namespace registration {
namespace internal {

// utils::concurrency::Mutex for all user registration operations.
extern utils::concurrency::Mutex registration_mutex;

struct Empty {};

template <class Interface, class... Args>
class AbstractCreator {
 public:
  virtual ~AbstractCreator() = default;
  AbstractCreator(absl::string_view name) : name_(name) {}
  const std::string& name() const { return name_; }
  virtual std::unique_ptr<Interface> Create(Args... args) = 0;

 private:
  std::string name_;
};

template <class Interface, class Implementation, class... Args>
class Creator final : public AbstractCreator<Interface, Args...> {
 public:
  Creator(absl::string_view name) : AbstractCreator<Interface, Args...>(name) {}
  std::unique_ptr<Interface> Create(Args... args) override {
    return absl::make_unique<Implementation>(args...);
  };
};

template <class Interface, class... Args>
class ClassPool {
 public:
  static std::vector<std::unique_ptr<AbstractCreator<Interface, Args...>>>*
  InternalGetItems() {
    static std::vector<std::unique_ptr<AbstractCreator<Interface, Args...>>>
        items;
    return &items;
  }

  static std::vector<std::string> GetNames() {
    utils::concurrency::MutexLock l(&registration_mutex);
    return InternalGetNames();
  }

  static bool IsName(absl::string_view name) {
    utils::concurrency::MutexLock l(&registration_mutex);
    auto& items = *InternalGetItems();
    for (const auto& item : items) {
      if (name == item->name()) {
        return true;
      }
    }
    return false;
  }

  static utils::StatusOr<std::unique_ptr<Interface>> Create(
      absl::string_view name, Args... args) {
    utils::concurrency::MutexLock l(&registration_mutex);
    auto& items = *InternalGetItems();
    for (const auto& item : items) {
      if (name != item->name()) {
        continue;
      }
      return item->Create(args...);
    }
    return absl::InvalidArgumentError(absl::Substitute(
        "Unknown item $0 in class pool $1. Registered elements are $2", name,
        typeid(Interface).name(), absl::StrJoin(InternalGetNames(), ",")));
  }

 private:
  static std::vector<std::string> InternalGetNames() {
    std::vector<std::string> names;
    auto& items = *InternalGetItems();
    for (const auto& item : items) {
      names.push_back(item->name());
    }
    return names;
  }
};

}  // namespace internal
}  // namespace registration
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_REGISTRATION_H_
