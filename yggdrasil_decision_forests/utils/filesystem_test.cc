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

#include "yggdrasil_decision_forests/utils/filesystem.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace file {
namespace {

using testing::ElementsAre;
using yggdrasil_decision_forests::test::DataRootDirectory;

TEST(Filesystem, JoinPath) {
  EXPECT_EQ(file::JoinPath("a", "b"), "a/b");
  EXPECT_EQ(file::JoinPath("a", "b", "c"), "a/b/c");
  EXPECT_EQ(file::JoinPath("/a", "b"), "/a/b");
}
TEST(Filesystem, GenerateShardedFilenames) {
  std::vector<std::string> shards;
  EXPECT_TRUE(GenerateShardedFilenames("b/a@3", &shards));
  EXPECT_THAT(shards, ElementsAre("b/a-00000-of-00003", "b/a-00001-of-00003",
                                  "b/a-00002-of-00003"));
}

TEST(Filesystem, Match) {
  std::vector<std::string> paths;
  EXPECT_OK(file::Match(
      file::JoinPath(DataRootDirectory(),
                     "yggdrasil_decision_forests/test_data/dataset/"
                     "toy.tfe-tfrecord*"),
      &paths, file::Defaults()));
  EXPECT_THAT(
      paths, ElementsAre(
                 file::JoinPath(
                     DataRootDirectory(),
                     "yggdrasil_decision_forests/test_data/dataset/"
                     "toy.tfe-tfrecord-00000-of-00002"),
                 file::JoinPath(
                     DataRootDirectory(),
                     "yggdrasil_decision_forests/test_data/dataset/"
                     "toy.tfe-tfrecord-00001-of-00002")));
}

TEST(Filesystem, FileIO) {
  auto tmp_dir = yggdrasil_decision_forests::test::TmpDirectory();
  auto file_path = JoinPath(tmp_dir, "my_file.txt");
  const std::string content = "hello world";
  auto output_handle = OpenOutputFile(file_path).value();
  EXPECT_OK(output_handle->Write(content));
  EXPECT_OK(output_handle->Close());

  auto input_handle = OpenInputFile(file_path).value();
  auto read_content = input_handle->ReadAll().value();
  EXPECT_OK(input_handle->Close());
  EXPECT_EQ(content, read_content);

  // Re-open the same file.
  EXPECT_OK(input_handle->Open(file_path));
  auto read_content_again = input_handle->ReadAll().value();
  EXPECT_OK(input_handle->Close());
  EXPECT_EQ(content, read_content_again);
}

TEST(Filesystem, Override) {
  auto file_path =
      JoinPath(yggdrasil_decision_forests::test::TmpDirectory(), "file.txt");
  EXPECT_OK(SetContent(file_path, "a"));
  EXPECT_EQ(GetContent(file_path).value(), "a");

  EXPECT_OK(SetContent(file_path, "b"));
  EXPECT_EQ(GetContent(file_path).value(), "b");
}

TEST(Filesystem, LargeContent) {
  auto file_path =
      JoinPath(yggdrasil_decision_forests::test::TmpDirectory(), "file.txt");
  std::string content(1024 * 10 + 512 + 64, 0);
  for (int i = 0; i < content.size(); i++) {
    content[i] = i % 255;
  }
  EXPECT_OK(SetContent(file_path, content));
  EXPECT_EQ(GetContent(file_path).value(), content);
}

TEST(Filesystem, ReadExactly) {
  auto tmp_dir = yggdrasil_decision_forests::test::TmpDirectory();
  auto file_path = JoinPath(tmp_dir, "my_file.txt");

  const std::string content = "01234567";  // 8 bytes
  auto output_handle = OpenOutputFile(file_path).value();
  EXPECT_OK(output_handle->Write(content));
  EXPECT_OK(output_handle->Close());

  char buffer[1024];

  {
    auto input_handle = OpenInputFile(file_path).value();
    EXPECT_TRUE(input_handle->ReadExactly(buffer, 4).value());
    EXPECT_EQ(std::memcmp(buffer, "0123", 4), 0);
    EXPECT_TRUE(input_handle->ReadExactly(buffer, 4).value());
    EXPECT_EQ(std::memcmp(buffer, "4567", 4), 0);
    EXPECT_FALSE(input_handle->ReadExactly(buffer, 4).value());
    EXPECT_OK(input_handle->Close());
  }

  {
    auto input_handle = OpenInputFile(file_path).value();
    EXPECT_FALSE(input_handle->ReadExactly(buffer, 9).ok());
    EXPECT_OK(input_handle->Close());
  }
}

TEST(Filesystem, FileIOHelper) {
  auto tmp_dir = yggdrasil_decision_forests::test::TmpDirectory();
  auto file_path = JoinPath(tmp_dir, "my_file.txt");
  const std::string content = "hello world";
  EXPECT_OK(SetContent(file_path, content));
  auto read_content = GetContent(file_path).value();
  EXPECT_EQ(content, read_content);
}

TEST(Filesystem, ProtoIO) {
  auto tmp_dir = yggdrasil_decision_forests::test::TmpDirectory();
  auto file_path = JoinPath(tmp_dir, "my_file.txt");
  yggdrasil_decision_forests::utils::proto::IntegerDistributionInt64 m1, m2;
  m1.set_sum(5);
  EXPECT_OK(SetBinaryProto(file_path, m1, file::Defaults()));
  EXPECT_OK(GetBinaryProto(file_path, &m2, file::Defaults()));
  EXPECT_EQ(m2.sum(), 5);
}

TEST(Filesystem, RecursivelyCreateDir) {
  auto tmp_dir = yggdrasil_decision_forests::test::TmpDirectory();
  EXPECT_OK(RecursivelyCreateDir(JoinPath(tmp_dir, "a", "b", "c"), Defaults()));
  EXPECT_OK(SetContent(JoinPath(tmp_dir, "a", "b", "c", "d.txt"), "hello"));
}

TEST(Filesystem, Rename) {
  auto tmp_dir = yggdrasil_decision_forests::test::TmpDirectory();
  auto file_a = JoinPath(tmp_dir, "a.txt");
  auto file_b = JoinPath(tmp_dir, "b.txt");
  const std::string content = "hello world";
  EXPECT_OK(SetContent(file_a, content));
  EXPECT_OK(Rename(file_a, file_b, Defaults()));
  auto read_content = GetContent(file_b).value();
  EXPECT_EQ(content, read_content);
}

TEST(Filesystem, Basename) {
  EXPECT_EQ(GetBasename("file"), "file");
  EXPECT_EQ(GetBasename("file.txt"), "file.txt");
  EXPECT_EQ(GetBasename(""), "");
  EXPECT_EQ(GetBasename(file::JoinPath("dir", "file")), "file");
  EXPECT_EQ(GetBasename(file::JoinPath("dir", "file.txt")), "file.txt");
  std::string dir_with_subdir = file::JoinPath("dir", "subdir");
  EXPECT_EQ(GetBasename(file::JoinPath(dir_with_subdir, "file")), "file");
  EXPECT_EQ(GetBasename(file::JoinPath(dir_with_subdir, "file.txt")),
            "file.txt");
}

}  // namespace
}  // namespace file
