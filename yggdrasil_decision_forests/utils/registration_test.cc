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

#include "yggdrasil_decision_forests/utils/registration.h"

#include <memory>
#include <string>
#include <type_traits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace registration {
namespace {

using ::yggdrasil_decision_forests::test::StatusIs;

class BaseClassA {
 public:
  virtual ~BaseClassA() = default;
  virtual std::string Result() = 0;
};

class SubClassA1 : public BaseClassA {
 public:
  SubClassA1(absl::string_view name) : name_(name) {}
  std::string Result() override { return "SubClassA1" + name_; }

 private:
  std::string name_;
};

class SubClassA2 : public BaseClassA {
 public:
  SubClassA2(absl::string_view name) : name_(name) {}
  std::string Result() override { return "SubClassA2" + name_; }

 private:
  std::string name_;
};

REGISTRATION_CREATE_POOL(BaseClassA, absl::string_view);
REGISTRATION_REGISTER_CLASS(SubClassA1, "A1", BaseClassA);
REGISTRATION_REGISTER_CLASS(SubClassA2, "A2", BaseClassA);

class BaseClassB {
 public:
  virtual ~BaseClassB() = default;
  virtual std::string Result() = 0;
};

class SubClassB1 : public BaseClassB {
 public:
  std::string Result() override { return "SubClassB1"; }
};

REGISTRATION_CREATE_POOL(BaseClassB);
REGISTRATION_REGISTER_CLASS(SubClassB1, "B1", BaseClassB);

class BaseClassC {
 public:
  using REQUIRED_REGISTRATION_CREATE = std::true_type;

  virtual ~BaseClassC() = default;
  virtual std::string Result() = 0;
};

class SubClassC1 : public BaseClassC {
 public:
  // A constructor with some complex args that we don't provide to the
  // registration.
  SubClassC1(int p1, int p2) {}

  static absl::StatusOr<std::unique_ptr<BaseClassC>> RegistrationCreate() {
    return absl::make_unique<SubClassC1>(1, 2);
  }

  std::string Result() override { return "SubClassC1"; }
};

REGISTRATION_CREATE_POOL(BaseClassC);
REGISTRATION_REGISTER_CLASS(SubClassC1, "C1", BaseClassC);

TEST(Registration, WithStatus) {
  EXPECT_EQ(BaseClassCRegisterer::Create("C1").value()->Result(), "SubClassC1");
}

TEST(Registration, WithConstructorArguments) {
  EXPECT_EQ(absl::StrJoin(BaseClassARegisterer::GetNames(), ","), "A1,A2");

  EXPECT_TRUE(BaseClassARegisterer::IsName("A1"));
  EXPECT_TRUE(BaseClassARegisterer::IsName("A2"));
  EXPECT_FALSE(BaseClassARegisterer::IsName("A3"));

  EXPECT_EQ(BaseClassARegisterer::Create("A1", "Toto").value()->Result(),
            "SubClassA1Toto");
  EXPECT_EQ(BaseClassARegisterer::Create("A1", "Titi").value()->Result(),
            "SubClassA1Titi");
  EXPECT_EQ(BaseClassARegisterer::Create("A2", "Tata").value()->Result(),
            "SubClassA2Tata");
  EXPECT_THAT(
      BaseClassARegisterer::Create("A3", "Tata").status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "No class registered with key \"A3\" in the class pool"));
}

TEST(Registration, WithOutConstructorArguments) {
  EXPECT_EQ(absl::StrJoin(BaseClassBRegisterer::GetNames(), ","), "B1");
  EXPECT_TRUE(BaseClassBRegisterer::IsName("B1"));
  EXPECT_FALSE(BaseClassBRegisterer::IsName("B3"));
  EXPECT_EQ(BaseClassBRegisterer::Create("B1").value()->Result(), "SubClassB1");
}

TEST(Registration, Threading) {
  utils::concurrency::ThreadPool pool(10);
  for (int i = 0; i < 10; ++i) {
    pool.Schedule([]() {
      for (int j = 0; j < 100; ++j) {
        EXPECT_EQ(BaseClassARegisterer::Create("A1", "T").value()->Result(),
                  "SubClassA1T");
        EXPECT_EQ(BaseClassBRegisterer::Create("B1").value()->Result(),
                  "SubClassB1");
        EXPECT_EQ(BaseClassCRegisterer::Create("C1").value()->Result(),
                  "SubClassC1");
      }
    });
  }
}

// Global thread pool to keep threads running during program exit.
utils::concurrency::ThreadPool* global_pool = nullptr;

class RegistrationTsanTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    global_pool = new utils::concurrency::ThreadPool(10);
    for (int i = 0; i < 10; ++i) {
      global_pool->Schedule([]() {
        // Keep accessing the registry indefinitely.
        while (true) {
          (void)BaseClassARegisterer::Create("A1", "T").value()->Result();
          (void)BaseClassBRegisterer::Create("B1").value()->Result();
          (void)BaseClassCRegisterer::Create("C1").value()->Result();
          // Yield to other threads.
          absl::SleepFor(absl::Milliseconds(1));
        }
      });
    }
  }

  static void TearDownTestSuite() {
    // Intentionally do not delete global_pool to keep threads running during
    // exit.
  }
};

TEST_F(RegistrationTsanTest, RunDuringExit) {
  // Give the threads in the global pool some time to run.
  absl::SleepFor(absl::Milliseconds(200));
  // When this test finishes, the test suite will tear down, but the
  // global_pool threads continue to run. Program exit will then occur,
  // potentially triggering the race if static destructors run while
  // threads are still accessing the registry.
}

}  // namespace
}  // namespace registration
}  // namespace yggdrasil_decision_forests
