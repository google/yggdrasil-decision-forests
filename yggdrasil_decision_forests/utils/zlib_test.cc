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

#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests::utils {
namespace {

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

}  // namespace
}  // namespace yggdrasil_decision_forests::utils
