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

// Unit tests of GCS filesystem.
//
// This test needs to be run with internet access and on a real cloud project.
// Usage example:
/*
Create find an existing a GC project and update "kTestBucket" bellow.

gcloud auth application-default login
gcloud config set project <project id>
bazel build \
  //yggdrasil_decision_forests/utils:filesystem_default_gcs_test \
  && \
bazel-bin/yggdrasil_decision_forests/utils/filesystem_default_gcs_test\
  --alsologtostderr
*/

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/utils/filesystem_default.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/uid.h"

// Name of a GCS bucket you have write and read access to.
constexpr char kTestBucket[] = "my_bucket_12498782345";

namespace yggdrasil_decision_forests::utils::filesystem {
namespace {

using ::testing::TestWithParam;
using ::yggdrasil_decision_forests::test::StatusIs;

TEST(GCSPath, Parse) {
  EXPECT_FALSE(GCSPath::Parse("").has_value());
  EXPECT_EQ(GCSPath::Parse("gs://my_bucket/a.txt").value(),
            GCSPath("my_bucket", "a.txt"));
}

TEST(Filesystem, WritingAndReadingFiles) {
  auto file_path = absl::StrCat("gs://", kTestBucket, "/dir/dir/test.txt");
  const std::string content = "hello world";

  ::file::FileOutputByteStream output_handle;
  EXPECT_OK(output_handle.Open(file_path));
  EXPECT_OK(output_handle.Write(content));
  EXPECT_OK(output_handle.Close());

  ::file::FileInputByteStream input_handle;
  EXPECT_OK(input_handle.Open(file_path));
  auto read_content = input_handle.ReadAll().value();
  EXPECT_OK(input_handle.Close());
  EXPECT_EQ(content, read_content);
}

TEST(Filesystem, ReadingNonExistingFile) {
  const auto uuid = ::yggdrasil_decision_forests::utils::GenUniqueId();
  const std::string path =
      absl::StrCat("gs://", kTestBucket, "/", uuid, "/file.txt");

  ::file::FileInputByteStream input_handle;

  EXPECT_THAT(input_handle.Open(path), StatusIs(absl::StatusCode::kUnknown));
}

TEST(Filesystem, Rename) {
  auto file_a = absl::StrCat("gs://", kTestBucket, "/a.txt");
  auto file_b = absl::StrCat("gs://", kTestBucket, "/b.txt");
  const std::string content = "hello world";

  ::file::FileOutputByteStream output_handle;
  EXPECT_OK(output_handle.Open(file_a));
  EXPECT_OK(output_handle.Write(content));
  EXPECT_OK(output_handle.Close());

  EXPECT_OK(::file::Rename(file_a, file_b, ::file::Defaults()));

  ::file::FileInputByteStream input_handle;
  EXPECT_OK(input_handle.Open(file_b));
  const std::string read_content = input_handle.ReadAll().value();
  EXPECT_OK(input_handle.Close());
  EXPECT_EQ(content, read_content);
}

// Note: The GCS API does not allow to rename dirs.
TEST(Filesystem, DISABLED_RenameDir) {
  const auto uuid1 = ::yggdrasil_decision_forests::utils::GenUniqueId();
  const auto uuid2 = ::yggdrasil_decision_forests::utils::GenUniqueId();

  auto dir_a = absl::StrCat("gs://", kTestBucket, "/", uuid1);
  auto dir_b = absl::StrCat("gs://", kTestBucket, "/", uuid2);
  auto file_a = absl::StrCat(dir_a, "/a.txt");
  auto file_b = absl::StrCat(dir_b, "/b.txt");
  const std::string content = "hello world";

  ::file::FileOutputByteStream output_handle;
  EXPECT_OK(output_handle.Open(file_a));
  EXPECT_OK(output_handle.Write(content));
  EXPECT_OK(output_handle.Close());

  EXPECT_OK(::file::Rename(dir_a, dir_b, ::file::Defaults()));

  ::file::FileInputByteStream input_handle;
  EXPECT_OK(input_handle.Open(file_b));
  const std::string read_content = input_handle.ReadAll().value();
  EXPECT_OK(input_handle.Close());
  EXPECT_EQ(content, read_content);
}

TEST(Filesystem, FileExistsAndDelete) {
  const auto uuid = ::yggdrasil_decision_forests::utils::GenUniqueId();
  const std::string path =
      absl::StrCat("gs://", kTestBucket, "/", uuid, "/file.txt");

  EXPECT_FALSE(::file::FileExists(path).value());

  ::file::FileOutputByteStream output_handle;
  EXPECT_OK(output_handle.Open(path));
  EXPECT_OK(output_handle.Write("hello world"));
  EXPECT_OK(output_handle.Close());

  EXPECT_TRUE(::file::FileExists(path).value());
  EXPECT_OK(::file::RecursivelyDelete(path, ::file::Defaults()));
  EXPECT_FALSE(::file::FileExists(path).value());
}

TEST(Filesystem, DeleteDirectory) {
  const auto uuid = ::yggdrasil_decision_forests::utils::GenUniqueId();
  const std::string dir =
      absl::StrCat("gs://", kTestBucket, "/", uuid, "/", uuid);
  const std::string file = absl::StrCat(dir, "/file.txt");

  EXPECT_FALSE(::file::FileExists(file).value());

  ::file::FileOutputByteStream output_handle;
  EXPECT_OK(output_handle.Open(file));
  EXPECT_OK(output_handle.Write("hello world"));
  EXPECT_OK(output_handle.Close());

  EXPECT_TRUE(::file::FileExists(file).value());
  EXPECT_OK(::file::RecursivelyDelete(dir, ::file::Defaults()));
  EXPECT_FALSE(::file::FileExists(file).value());
}

struct MatchTestCase {
  std::string test_name;
  std::vector<std::string> available;
  std::vector<std::string> expected;
  std::string pattern;
};

using MatchTest = TestWithParam<MatchTestCase>;

TEST_P(MatchTest, Basic) {
  const MatchTestCase& test_case = GetParam();
  const auto uuid = ::yggdrasil_decision_forests::utils::GenUniqueId();
  const std::string dir = absl::StrCat("gs://", kTestBucket, "/", uuid, "/");

  for (const std::string filename : test_case.available) {
    ::file::FileOutputByteStream output_handle;
    EXPECT_OK(output_handle.Open(absl::StrCat(dir, filename)));
    EXPECT_OK(output_handle.Write("something"));
    EXPECT_OK(output_handle.Close());
  }

  std::vector<std::string> paths;
  EXPECT_OK(::file::Match(absl::StrCat(dir, test_case.pattern), &paths,
                          ::file::Defaults()));

  std::vector<std::string> expected_with_bucket;
  for (const auto& exp : test_case.expected) {
    expected_with_bucket.push_back(
        absl::StrCat("gs://", kTestBucket, "/", uuid, "/", exp));
  }
  EXPECT_EQ(paths, expected_with_bucket);
}

INSTANTIATE_TEST_SUITE_P(
    MatchTestInstantiation, MatchTest,
    testing::ValuesIn<MatchTestCase>({
        {
            .test_name = "prefix_digit",
            .available = {"1hello", "2hello", "3hello", "4the", "5world"},
            .expected = {"1hello", "2hello", "3hello"},
            .pattern = "*hello",
        },
        {
            .test_name = "post_digit",
            .available = {"hello1", "hello2", "hello3", "the4", "world5"},
            .expected = {"hello1", "hello2", "hello3"},
            .pattern = "hello*",
        },
        {
            .test_name = "question_mark",
            .available =
                {
                    "hello10",
                    "hello11",
                    "hello20",
                    "hello21",
                    "hello30",
                    "world11",
                },
            .expected = {"hello11", "hello21"},
            .pattern = "hello?1",
        },
        {
            .test_name = "bracket",
            .available = {"helloA", "helloB", "helloC", "helloD"},
            .expected = {"helloA", "helloC"},
            .pattern = "hello[AC]",
        },
    }),
    [](const testing::TestParamInfo<MatchTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace yggdrasil_decision_forests::utils::filesystem
