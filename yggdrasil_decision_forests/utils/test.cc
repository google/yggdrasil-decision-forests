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

#include <algorithm>
#include <cstddef>
#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"  // Required for diffing
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"
#include "yggdrasil_decision_forests/utils/uid.h"  // IWYU pragma: keep

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
  // Setup directory containing results details.
  const std::string output_dir = file::JoinPath(TmpDirectory(), "golden");
  ASSERT_OK(file::RecursivelyCreateDir(output_dir, file::Defaults()));
  const std::string all_commands_path =
      file::JoinPath(output_dir, absl::StrCat("all_commands.txt"));

  // Call this method to add a line to the "all_commands.txt" file generated in
  // the output directory.
  const auto add_to_all_commands = [&all_commands_path](
                                       absl::string_view command) {
    static int first = true;
    std::string all_commands;
    if (!first) {
      ASSERT_OK_AND_ASSIGN(all_commands, file::GetContent(all_commands_path));
    }
    LOG(INFO) << "New command added:\n" << all_commands_path;
    absl::StrAppend(&all_commands, command, "\n");
    ASSERT_OK(file::SetContent(all_commands_path, all_commands));
    first = false;
  };

  // Check the existence of the golden file.
  const std::string full_path = file::JoinPath(DataRootDirectory(), path);
  ASSERT_OK_AND_ASSIGN(const bool golden_file_exists,
                       file::FileExists(full_path));
  if (!golden_file_exists) {
    LOG(INFO) << "The following golden file does not exist:\n" << path;
    add_to_all_commands(absl::StrCat("touch ", path));
    EXPECT_TRUE(false);
  }

  ASSERT_OK_AND_ASSIGN(auto expected_content, file::GetContent(full_path));

  for (const auto& token : tokens_to_replace) {
    std::regex token_regex(absl::StrCat(R"(\$\{)", token.first, R"(\})"));
    expected_content =
        std::regex_replace(expected_content, token_regex, token.second);
  }

  if (expected_content != content) {
    LOG(INFO) << "The given value does not match the golden value: " << path;

    std::vector<absl::string_view> expected_lines =
        absl::StrSplit(expected_content, '\n');
    std::vector<absl::string_view> actual_lines = absl::StrSplit(content, '\n');

    const size_t n = expected_lines.size();
    const int m = actual_lines.size();

    // Compute LCS table
    std::vector<std::vector<int>> lcs_table(n + 1, std::vector<int>(m + 1, 0));

    for (int i = 1; i <= n; ++i) {
      for (int j = 1; j <= m; ++j) {
        if (expected_lines[i - 1] == actual_lines[j - 1]) {
          lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1;
        } else {
          lcs_table[i][j] = std::max(lcs_table[i - 1][j], lcs_table[i][j - 1]);
        }
      }
    }

    // Backtrack to find the diff trace
    struct DiffStep {
      char type;  // ' ' = Match, '-' = Deletion, '+' = Addition
      absl::string_view line;
    };
    std::vector<DiffStep> trace;
    int i = n, j = m;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && expected_lines[i - 1] == actual_lines[j - 1]) {
        // Lines match.
        trace.push_back({' ', expected_lines[i - 1]});
        i--;
        j--;
      } else if (j > 0 &&
                 (i == 0 || lcs_table[i][j - 1] >= lcs_table[i - 1][j])) {
        // Addition.
        trace.push_back({'+', actual_lines[j - 1]});
        j--;
      } else {
        // Deletion.
        trace.push_back({'-', expected_lines[i - 1]});
        i--;
      }
    }
    std::reverse(trace.begin(), trace.end());

    // Determine what to print with context.
    std::vector<bool> should_print(trace.size(), false);
    const int context_lines = 2;

    for (int k = 0; k < trace.size(); ++k) {
      if (trace[k].type != ' ') {
        int start = std::max(0, k - context_lines);
        int end =
            std::min(static_cast<int>(trace.size()) - 1, k + context_lines);
        for (int c = start; c <= end; ++c) {
          should_print[c] = true;
        }
      }
    }

    LOG(INFO) << "Diff:";
    LOG(INFO) << "==========================================";

    for (int k = 0; k < trace.size(); ++k) {
      if (should_print[k]) {
        if (k > 0 && !should_print[k - 1]) {
          LOG(INFO) << "@@ ... @@";
        }
        LOG(INFO) << trace[k].type << " " << trace[k].line;
      }
    }
    LOG(INFO) << "==========================================";
    // --- Diff Logic End ---

    static int actual_idx = 0;
    const std::string output_path = file::JoinPath(
        output_dir, absl::StrCat("actual_", actual_idx, ".html"));
    const std::string expected_output_path = file::JoinPath(
        output_dir, absl::StrCat("expected_", actual_idx, ".html"));

    LOG(INFO) << "Content saved to " << output_path;
    LOG(INFO) << "";
    const std::string cp_command = absl::StrCat("cp ", output_path, " ", path);
    add_to_all_commands(cp_command);
    LOG(INFO) << "Update the golden file with:\n" << cp_command;
    LOG(INFO) << "";
    LOG(INFO) << "Look at the difference between the fields with:\ndiff "
              << output_path << " " << path;
    LOG(INFO) << "";
    LOG(INFO) << "Expected: " << expected_output_path;
    ASSERT_OK(file::SetContent(expected_output_path, expected_content));
    ASSERT_OK(file::SetContent(output_path, content));

    EXPECT_TRUE(false);
    actual_idx++;
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
