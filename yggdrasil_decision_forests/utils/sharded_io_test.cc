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

#include "yggdrasil_decision_forests/utils/sharded_io.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

using testing::ElementsAre;

std::string TestDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

TEST(ShardedIO, ShardedReader) {
  class TestShardedReader : public ShardedReader<std::string> {
   public:
    ~TestShardedReader() override = default;
    absl::Status OpenShard(absl::string_view path) override {
      last_path_ = std::string(path);
      next_value_ = 0;
      return absl::OkStatus();
    }
    utils::StatusOr<bool> NextInShard(std::string* value) override {
      if (next_value_ == 2) {
        return false;
      }
      value->clear();
      absl::StrAppend(value, last_path_, next_value_++);
      return true;
    }

   private:
    int next_value_ = 0;
    std::string last_path_;
  };

  TestShardedReader test;
  EXPECT_OK(test.Open({"a", "b"}));
  std::string value;
  EXPECT_TRUE(test.Next(&value).value());
  EXPECT_EQ(value, "a0");
  EXPECT_TRUE(test.Next(&value).value());
  EXPECT_EQ(value, "a1");
  EXPECT_TRUE(test.Next(&value).value());
  EXPECT_EQ(value, "b0");
  EXPECT_TRUE(test.Next(&value).value());
  EXPECT_EQ(value, "b1");
  EXPECT_FALSE(test.Next(&value).value());
}

TEST(ShardedIO, ShardedWriter) {
  class TestShardedWriter : public ShardedWriter<std::string> {
   public:
    ~TestShardedWriter() override = default;
    absl::Status CloseWithStatus() final { return absl::OkStatus(); }
    absl::Status OpenShard(absl::string_view path) final {
      absl::StrAppend(&logs_, " O[", path, "]");
      return absl::OkStatus();
    }
    absl::Status WriteInShard(const std::string& value) final {
      absl::StrAppend(&logs_, " W[", value, "]");
      return absl::OkStatus();
    }

   public:
    // Will contains a string representation of the operations.
    std::string logs_;

   private:
    int next_value_ = 0;
    std::string last_path_;
  };

  TestShardedWriter test;
  EXPECT_OK(test.Open("f@2", 2));
  EXPECT_OK(test.Write("a"));
  EXPECT_OK(test.Write("b"));
  EXPECT_OK(test.Write("c"));
  EXPECT_OK(test.Write("d"));

  EXPECT_EQ(test.logs_,
            " O[f-00000-of-00002] W[a] W[b] O[f-00001-of-00002] W[c] W[d]");
}

TEST(ShardedIO, ExpandInputShards) {
  // If EXTERNAL_FILESYSTEM is test, patterns on the directory name are not
  // tested.
  std::vector<std::string> paths;

  EXPECT_OK(ExpandInputShards(
      file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord-00000-of-00002"),
      &paths));
  EXPECT_THAT(paths,
              ElementsAre(file::JoinPath(
                  TestDir(), "dataset/toy.tfe-tfrecord-00000-of-00002")));

  EXPECT_OK(ExpandInputShards(
      file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord@2"), &paths));
  EXPECT_THAT(
      paths,
      ElementsAre(
          file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord-00000-of-00002"),
          file::JoinPath(TestDir(),
                         "dataset/toy.tfe-tfrecord-00001-of-00002")));

  EXPECT_OK(ExpandInputShards(
      file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord*"), &paths));
  EXPECT_THAT(
      paths,
      ElementsAre(
          file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord-00000-of-00002"),
          file::JoinPath(TestDir(),
                         "dataset/toy.tfe-tfrecord-00001-of-00002")));

#ifndef EXTERNAL_FILESYSTEM
  EXPECT_OK(ExpandInputShards(file::JoinPath(TestDir(), "*/toy.tfe-tfrecord*"),
                              &paths));
  EXPECT_THAT(
      paths,
      ElementsAre(
          file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord-00000-of-00002"),
          file::JoinPath(TestDir(),
                         "dataset/toy.tfe-tfrecord-00001-of-00002")));

  EXPECT_OK(ExpandInputShards(file::JoinPath(TestDir(), "*/toy.tfe-tfrecord@2"),
                              &paths));
  EXPECT_THAT(
      paths,
      ElementsAre(
          file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord-00000-of-00002"),
          file::JoinPath(TestDir(),
                         "dataset/toy.tfe-tfrecord-00001-of-00002")));
#endif

  EXPECT_OK(ExpandInputShards(
      absl::StrCat(
          file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord-00000-of-00002"),
          ",",
          file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord-00001-of-00002")),
      &paths));
  EXPECT_THAT(
      paths,
      ElementsAre(
          file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord-00000-of-00002"),
          file::JoinPath(TestDir(),
                         "dataset/toy.tfe-tfrecord-00001-of-00002")));

#ifndef EXTERNAL_FILESYSTEM
  EXPECT_OK(ExpandInputShards(
      absl::StrCat(
          file::JoinPath(TestDir(), "*/toy.tfe-tfrecord-00000-of-00002"), ",",
          file::JoinPath(TestDir(), "*/toy.tfe-tfrecord-00001-of-00002")),
      &paths));
  EXPECT_THAT(
      paths,
      ElementsAre(
          file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord-00000-of-00002"),
          file::JoinPath(TestDir(),
                         "dataset/toy.tfe-tfrecord-00001-of-00002")));

  EXPECT_OK(ExpandInputShards(
      absl::StrCat(file::JoinPath(TestDir(), "*/toy.tfe-tfrecord@2"), ",",
                   file::JoinPath(TestDir(), "*/toy.csv")),
      &paths));
  EXPECT_THAT(
      paths,
      ElementsAre(
          file::JoinPath(TestDir(), "dataset/toy.csv"),
          file::JoinPath(TestDir(), "dataset/toy.tfe-tfrecord-00000-of-00002"),
          file::JoinPath(TestDir(),
                         "dataset/toy.tfe-tfrecord-00001-of-00002")));
#endif
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
