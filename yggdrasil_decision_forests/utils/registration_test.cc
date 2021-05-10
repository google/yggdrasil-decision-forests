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

#include "yggdrasil_decision_forests/utils/registration.h"

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
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
  EXPECT_THAT(BaseClassARegisterer::Create("A3", "Tata").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unknown item A3 in class pool"));
}

TEST(Registration, WithOutConstructorArguments) {
  EXPECT_EQ(absl::StrJoin(BaseClassBRegisterer::GetNames(), ","), "B1");
  EXPECT_TRUE(BaseClassBRegisterer::IsName("B1"));
  EXPECT_FALSE(BaseClassBRegisterer::IsName("B3"));
  EXPECT_EQ(BaseClassBRegisterer::Create("B1").value()->Result(), "SubClassB1");
}

}  // namespace
}  // namespace registration
}  // namespace yggdrasil_decision_forests
