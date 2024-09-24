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

#include "yggdrasil_decision_forests/utils/zlib.h"

#include <stddef.h>

#include <memory>
#include <random>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests::utils {
namespace {

using ::testing::TestWithParam;
using yggdrasil_decision_forests::test::DataRootDirectory;

std::string HelloPath() {
  return file::JoinPath(
      DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/dataset/"
      "hello.txt.gz");
}

TEST(GZip, ReadAll) {
  auto file_stream = file::OpenInputFile(HelloPath()).value();
  auto stream = GZipInputByteStream::Create(std::move(file_stream)).value();
  auto read_content = stream->ReadAll().value();
  EXPECT_OK(stream->Close());
  EXPECT_EQ("hello", read_content);
}

TEST(GZip, ReadUnit) {
  auto file_stream = file::OpenInputFile(HelloPath()).value();
  auto stream = GZipInputByteStream::Create(std::move(file_stream)).value();
  char buffer[10];

  int n = stream->ReadUpTo(buffer, 1).value();
  EXPECT_EQ(n, 1);
  EXPECT_EQ(buffer[0], 'h');

  n = stream->ReadUpTo(buffer, 2).value();
  EXPECT_EQ(n, 2);
  EXPECT_EQ(buffer[0], 'e');
  EXPECT_EQ(buffer[1], 'l');

  n = stream->ReadUpTo(buffer, 5).value();
  EXPECT_EQ(n, 2);
  EXPECT_EQ(buffer[0], 'l');
  EXPECT_EQ(buffer[1], 'o');

  n = stream->ReadUpTo(buffer, 5).value();
  EXPECT_EQ(n, 0);

  EXPECT_OK(stream->Close());
}

struct GZipTestCase {
  size_t content_size;
  size_t buffer_size;
  int compression;
  bool random = true;
};

using GZipTestCaseTest = TestWithParam<GZipTestCase>;

INSTANTIATE_TEST_SUITE_P(
    GZipTestCaseTestSuiteInstantiation, GZipTestCaseTest,
    testing::ValuesIn<GZipTestCase>({{1024 * 2, 1024, 8},
                                     {10, 1024, 8},
                                     {10, 256, 8},
                                     {0, 1024 * 1024, 8},
                                     {10, 1024 * 1024, 8},
                                     {10 * 1024 * 1024, 1024 * 1024, 8},
                                     {10 * 1024 * 1024, 1024 * 1024, 8, false},
                                     {10 * 1024 * 1024, 1024 * 1024, 1},
                                     {10 * 1024 * 1024, 1024 * 1024, -1}}));

TEST_P(GZipTestCaseTest, WriteAndRead) {
  const GZipTestCase& test_case = GetParam();
  std::uniform_int_distribution<int> dist(0, 10);
  std::mt19937_64 rng(1);

  auto tmp_dir = test::TmpDirectory();
  auto file_path = file::JoinPath(tmp_dir, "my_file.txt.gz");
  std::string content;
  content.reserve(test_case.content_size);
  if (test_case.random) {
    for (int i = 0; i < test_case.content_size; i++) {
      absl::StrAppend(&content, dist(rng));
    }
  } else {
    for (int i = 0; i < test_case.content_size / 2; i++) {
      absl::StrAppend(&content, "AB");
    }
  }

  {
    auto file_stream = file::OpenOutputFile(file_path).value();
    auto stream = GZipOutputByteStream::Create(std::move(file_stream),
                                               test_case.compression,
                                               test_case.buffer_size)
                      .value();
    EXPECT_OK(stream->Write(content));
    EXPECT_OK(stream->Write(content));
    EXPECT_OK(stream->Close());
  }

  {
    auto file_stream = file::OpenInputFile(file_path).value();
    auto stream = GZipInputByteStream::Create(std::move(file_stream),
                                              test_case.buffer_size)
                      .value();
    auto read_content = stream->ReadAll().value();
    EXPECT_OK(stream->Close());
    EXPECT_EQ(content + content, read_content);
  }
}

}  // namespace
}  // namespace yggdrasil_decision_forests::utils
