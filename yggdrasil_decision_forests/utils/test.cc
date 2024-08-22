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

#include "yggdrasil_decision_forests/utils/test.h"

#include <random>
#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests {
namespace test {

std::string DataRootDirectory() { return ""; }

std::string TmpDirectory() {
  // Ensure that each test uses a separate test directory as tests can run
  // concurrently.
  const auto test = testing::UnitTest::GetInstance()->current_test_info();
  CHECK(test != nullptr);
  const auto test_name = absl::StrCat(
      test->test_suite_name(), "-", test->test_case_name(), "-",
      test->name(), "-", utils::GenUniqueId());
  std::string path =
      file::JoinPath(testing::TempDir(), "yggdrasil_unittest", test_name);
  CHECK_OK(file::RecursivelyCreateDir(path, file::Defaults()));
  return path;
}

void ExpectEqualGolden(
    absl::string_view content, absl::string_view path,
    const std::vector<std::pair<std::string, std::string>>& tokens_to_replace) {
  ASSERT_OK_AND_ASSIGN(auto expected_content, file::GetContent(file::JoinPath(
                                                  DataRootDirectory(), path)));
  for (const auto& token : tokens_to_replace) {
    std::regex token_regex(absl::StrCat(R"(\$\{)", token.first, R"(\})"));
    expected_content =
        std::regex_replace(expected_content, token_regex, token.second);
  }
  if (expected_content != content) {
    LOG(INFO) << "The given value does not match the golden value: " << path;

    const int max_print = 1000;
    if (content.size() < max_print) {
      LOG(INFO) << "Given value  (" << content.size()
                << " characters)\n====================\n"
                << content << "\n====================";
    } else {
      LOG(INFO)
          << "The content is too large (" << content.size()
          << " characters) to be printed.\nFirst part:\n====================\n"
          << content.substr(0, max_print) << "\n====================";
    }

    if (expected_content.size() < max_print) {
      LOG(INFO) << "Expected value  (" << expected_content.size()
                << " characters)\n====================\n"
                << expected_content << "\n====================";
    } else {
      LOG(INFO)
          << "The expected_content is too large (" << expected_content.size()
          << " characters) to be printed.\nFirst part:\n====================\n"
          << expected_content.substr(0, max_print) << "\n====================";
    }

    static int actual_idx = 0;
    const std::string output_dir = file::JoinPath(TmpDirectory(), "golden");
    ASSERT_OK(file::RecursivelyCreateDir(output_dir, file::Defaults()));
    const std::string output_path = file::JoinPath(
        output_dir, absl::StrCat("actual_", actual_idx, ".html"));
    const std::string expected_output_path = file::JoinPath(
        output_dir, absl::StrCat("expected_", actual_idx++, ".html"));
    LOG(INFO) << "Content saved to " << output_path;
    LOG(INFO) << "";
    LOG(INFO) << "Update the golden file with:\ncp " << output_path << " "
              << path;
    LOG(INFO) << "";
    LOG(INFO) << "Look at the difference between the fields with:\ndiff "
              << output_path << " " << path;
    LOG(INFO) << "";
    LOG(INFO) << "Expected: " << expected_output_path;
    ASSERT_OK(file::SetContent(expected_output_path, expected_content));
    ASSERT_OK(file::SetContent(output_path, content));
    EXPECT_TRUE(false);
  }
}

#ifdef _WIN32

bool IsPortAvailable(int port) {
  LOG(WARNING) << "Validating port " << port << " without checking it.";
  return true;
}

#else

#include <netinet/in.h>

// Copied from tensorflow/core/platform/default/net.cc.
bool IsPortAvailable(int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);
  int actual_port;

  if (fd < 0) {
    LOG(ERROR) << "socket() failed: " << strerror(errno);
    return false;
  }

  // SO_REUSEADDR lets us start up a server immediately after it exists.
  int one = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
    LOG(ERROR) << "setsockopt() failed: " << strerror(errno);
    if (close(fd) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    };
    return false;
  }

  // Try binding to port.
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0)
  {
    LOG(WARNING) << "bind(port=" << port << ") failed: " <<
    strerror(errno); if (close(fd) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    };
    return false;
  }

  // Get the bound port number.
  if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&addr), &addr_len) <
      0) {
    LOG(WARNING) << "getsockname() failed: " << strerror(errno);
    if (close(fd) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    };
    return false;
  }
  CHECK_LE(addr_len, sizeof(addr));
  actual_port = ntohs(addr.sin_port);
  CHECK_GT(actual_port, 0);

  CHECK_EQ(port, actual_port);
  if (close(fd) < 0) {
    LOG(ERROR) << "close() failed: " << strerror(errno);
  }

  return true;
}
#endif

int PickUnusedPortOrDie() {
  // Ephemeral ports.
  std::uniform_int_distribution<int> dist_port(49152, 65535);
  std::mt19937 rnd(std::random_device{}());

  for (int trial = 0; trial < 1000; trial++) {
    const int port = dist_port(rnd);
    if (IsPortAvailable(port)) {
      return port;
    }
  }
  LOG(WARNING) << "Failed to pick an unused port";
  return 0;
}

}  // namespace test
}  // namespace yggdrasil_decision_forests
