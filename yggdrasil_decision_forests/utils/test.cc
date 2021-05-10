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

#include "yggdrasil_decision_forests/utils/test.h"

#include "gtest/gtest.h"

#include "yggdrasil_decision_forests/utils/filesystem.h"

namespace yggdrasil_decision_forests {
namespace test {

std::string DataRootDirectory() { return ""; }

std::string TmpDirectory() {
  // Ensure that each test uses a separate test directory as tests can run
  // concurrently.
  const auto test = testing::UnitTest::GetInstance()->current_test_info();
  CHECK(test != nullptr);
  const auto test_name = absl::StrCat(
      test->test_suite_name(), "-", test->test_case_name(), "-", test->name());
  std::string path =
      file::JoinPath(testing::TempDir(), "yggdrasil_unittest", test_name);
  CHECK_OK(file::RecursivelyCreateDir(path, file::Defaults()));
  return path;
}

}  // namespace test
}  // namespace yggdrasil_decision_forests
