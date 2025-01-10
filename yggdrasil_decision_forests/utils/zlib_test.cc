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
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
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
  auto stream = GZipInputByteStream::Create(file_stream.get()).value();
  auto read_content = stream->ReadAll().value();
  EXPECT_OK(stream->Close());
  EXPECT_EQ("hello", read_content);
}

TEST(GZip, ReadUnit) {
  auto file_stream = file::OpenInputFile(HelloPath()).value();
  auto stream = GZipInputByteStream::Create(file_stream.get()).value();
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
    auto stream =
        GZipOutputByteStream::Create(file_stream.get(), test_case.compression,
                                     test_case.buffer_size)
            .value();
    EXPECT_OK(stream->Write(content));
    EXPECT_OK(stream->Write(content));
    EXPECT_OK(stream->Close());
    EXPECT_OK(file_stream->Close());
  }

  {
    auto file_stream = file::OpenInputFile(file_path).value();
    auto stream =
        GZipInputByteStream::Create(file_stream.get(), test_case.buffer_size)
            .value();
    auto read_content = stream->ReadAll().value();
    EXPECT_OK(stream->Close());
    EXPECT_OK(file_stream->Close());
    EXPECT_EQ(content + content, read_content);
  }
}

TEST(RawInflate, Base) {
  const auto input =
      absl::HexStringToBytes("05804109000008c4aa184ec1c7e0c08ff5c70ea43e470b");
  std::string output;
  std::string working_buffer(1024, 0);
  ASSERT_OK(Inflate(input, &output, &working_buffer, /*raw_deflate=*/true));
  EXPECT_EQ(output, "hello world");
}

TEST(RawInflate, ExceedBuffer) {
  // Create a large chunk of data (need to be larger than the
  // decompress buffer for this test to make sense).
  std::string raw_data;
  raw_data.reserve(13'000'000);
  // Write 13MB of non-compressed data.
  for (int i = 0; i < 1'000'000; i++) {
    absl::StrAppend(&raw_data, "13 characters");
  }

  std::string compressed_data;
  {
    // Compress the data.
    auto raw_stream = std::make_unique<StringOutputByteStream>();
    auto stream = GZipOutputByteStream::Create(raw_stream.get(), 8, 1024 * 1024,
                                               /*raw_deflate=*/true)
                      .value();
    EXPECT_OK(stream->Write(raw_data));
    EXPECT_OK(stream->Close());
    EXPECT_OK(raw_stream->Close());

    compressed_data = raw_stream->ToString();
    LOG(INFO) << "Compressed data size:" << compressed_data.size();
  }

  std::string output;
  // The buffer is smaller than the decompressed data.
  std::string working_buffer(1024 * 1024, 0);
  ASSERT_OK(
      Inflate(compressed_data, &output, &working_buffer, /*raw_deflate=*/true));
  EXPECT_EQ(output, raw_data);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::utils
