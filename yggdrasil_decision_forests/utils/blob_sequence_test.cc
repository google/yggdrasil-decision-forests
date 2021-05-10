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

#include "yggdrasil_decision_forests/utils/blob_sequence.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace blob_sequence {
namespace {

TEST(BlobSequence, Base) {
  auto path = file::JoinPath(test::TmpDirectory(), "blob_sequence.bin");

  // Create a BS with three blobs.
  auto output_stream = file::OpenOutputFile(path).value();
  auto writer = blob_sequence::Writer::Create(output_stream.get()).value();
  CHECK_OK(writer.Write("HELLO"));
  CHECK_OK(writer.Write(""));  // Empty blob.
  CHECK_OK(writer.Write("WORLD"));
  CHECK_OK(writer.Close());
  CHECK_OK(output_stream->Close());

  // Read the two blobs.
  auto input_stream = file::OpenInputFile(path).value();
  auto reader = blob_sequence::Reader::Create(input_stream.get()).value();
  std::string blob;
  CHECK(reader.Read(&blob).value());
  CHECK_EQ(blob, "HELLO");
  CHECK(reader.Read(&blob).value());
  CHECK_EQ(blob, "");
  CHECK(reader.Read(&blob).value());
  CHECK_EQ(blob, "WORLD");
  CHECK(!reader.Read(&blob).value());
  CHECK_OK(reader.Close());
  CHECK_OK(input_stream->Close());
}

TEST(BlobSequence, NotABs) {
  auto path = file::JoinPath(test::TmpDirectory(), "blob_sequence.bin");

  // Create a random text file (not a BS file).
  auto output_stream = file::OpenOutputFile(path).value();
  CHECK_OK(output_stream->Write("HELLO WORLD"));
  CHECK_OK(output_stream->Close());

  // Try to read the file.
  auto input_stream = file::OpenInputFile(path).value();
  EXPECT_THAT(blob_sequence::Reader::Create(input_stream.get()).status(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));
  CHECK_OK(input_stream->Close());
}

}  // namespace
}  // namespace blob_sequence
}  // namespace utils
}  // namespace yggdrasil_decision_forests
