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

#include "yggdrasil_decision_forests/utils/csv.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace csv {
namespace {

using testing::ElementsAre;

std::string DatasetDir() {
  return file::JoinPath(
      test::DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/dataset");
}

TEST(Csv, ReadToy) {
  auto file_handle =
      file::OpenInputFile(file::JoinPath(DatasetDir(), "toy.csv")).value();
  Reader reader(file_handle.get());
  std::vector<absl::string_view>* row;

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("Num_1", "Num_2", "Cat_1", "Cat_2", "Cat_set_1",
                                "Cat_set_2", "Bool_1", "Bool_2", "Cat_3"));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("1", "NA", "A", "A", "X", "", "0", "0", "1"));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("2", "2", "B", "", "X Y", "X", "1", "NA", "2"));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row,
              ElementsAre("3", "NA", "A", "B", "Y X Z", "X Y", "0", "1", "1"));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row,
              ElementsAre("4", "4", "C", "", "X Y Z", "Z X Y", "1", "NA", "3"));

  EXPECT_FALSE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre());
  EXPECT_OK(file_handle->Close());
}

TEST(Csv, UnixNL) {
  StringInputByteStream stream("a,b\nc,d\n");
  Reader reader(&stream);
  std::vector<absl::string_view>* row;

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("a", "b"));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("c", "d"));

  EXPECT_FALSE(reader.NextRow(&row).value());
}

TEST(Csv, WindowsNL) {
  StringInputByteStream stream("a,b\r\nc,d\r\n");
  Reader reader(&stream);
  std::vector<absl::string_view>* row;

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("a", "b"));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("c", "d"));

  EXPECT_FALSE(reader.NextRow(&row).value());
}

TEST(Csv, MacNL) {
  StringInputByteStream stream("a,b\rc,d\r");
  Reader reader(&stream);
  std::vector<absl::string_view>* row;

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("a", "b"));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("c", "d"));

  EXPECT_FALSE(reader.NextRow(&row).value());
}

TEST(Csv, Empties) {
  StringInputByteStream stream(R"(a,b
,,

,
)");
  Reader reader(&stream);
  std::vector<absl::string_view>* row;

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("a", "b"));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("", "", ""));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre(""));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("", ""));

  EXPECT_FALSE(reader.NextRow(&row).value());
}

TEST(Csv, Quotes) {
  StringInputByteStream stream(R"(a,b
"c","d"
"",""
"""",""""
"a""b",",c
d")");
  Reader reader(&stream);
  std::vector<absl::string_view>* row;

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("a", "b"));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("c", "d"));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("", ""));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("\"", "\""));

  EXPECT_TRUE(reader.NextRow(&row).value());
  EXPECT_THAT(*row, ElementsAre("a\"b", ",c\nd"));

  EXPECT_FALSE(reader.NextRow(&row).value());
}

TEST(Csv, EOFOpenQuote) {
  StringInputByteStream stream(R"(a,"b
c,d
)");
  Reader reader(&stream);
  std::vector<absl::string_view>* row;
  EXPECT_FALSE(reader.NextRow(&row).ok());
}

TEST(Csv, EOFHalfQuote) {
  StringInputByteStream stream(R"(a,b""c)");
  Reader reader(&stream);
  std::vector<absl::string_view>* row;
  EXPECT_FALSE(reader.NextRow(&row).ok());
}

TEST(Csv, WriterWindows) {
  const auto file_path = file::JoinPath(test::TmpDirectory(), "my_file.csv");
  auto output_handle = file::OpenOutputFile(file_path).value();

  // Write some content.
  const std::vector<std::vector<absl::string_view>> content = {
      {"abc", "123"},   // Basic
      {{}},             // One empty field (empty row cannot be represented)
      {{}, {}, {}},     // Empty fields
      {"hello world"},  // With space
      {"a\"b\"c,xyz"},  // With quote
      {"a\nb\nc,xyz"}   // With line return
  };
  Writer writer(output_handle.get(), NewLine::WINDOWS);
  for (const auto& row : content) {
    EXPECT_OK(writer.WriteRow(row));
  }
  EXPECT_OK(output_handle->Close());

  // Check the writen bytes.
  auto input_handle = file::OpenInputFile(file_path).value();
  auto read_content = input_handle->ReadAll().value();
  EXPECT_OK(input_handle->Close());
  EXPECT_EQ(
      "abc,123\r\n\r\n,,\r\nhello "
      "world\r\n\"a\"\"b\"\"c,xyz\"\r\n\"a\nb\nc,xyz\"\r\n",
      read_content);

  // Read back the content with the csv reader.
  input_handle = file::OpenInputFile(file_path).value();
  Reader reader(input_handle.get());
  std::vector<absl::string_view>* row;
  int row_idx = 0;
  while (reader.NextRow(&row).value()) {
    LOG(INFO) << "row_idx:" << row_idx;
    EXPECT_EQ(*row, content[row_idx]);
    row_idx++;
  }
  EXPECT_EQ(row_idx, content.size());
  EXPECT_OK(input_handle->Close());
}

TEST(Csv, WriterUnix) {
  const auto file_path = file::JoinPath(test::TmpDirectory(), "my_file.csv");
  auto output_handle = file::OpenOutputFile(file_path).value();

  // Write some content.
  const std::vector<std::vector<absl::string_view>> content = {
      {"abc", "123"},   // Basic
      {{}},             // One empty field (empty row cannot be represented)
      {{}, {}, {}},     // Empty fields
      {"hello world"},  // With space
      {"a\"b\"c,xyz"},  // With quote
      {"a\nb\nc,xyz"}   // With line return
  };
  Writer writer(output_handle.get(), NewLine::UNIX);
  for (const auto& row : content) {
    EXPECT_OK(writer.WriteRow(row));
  }
  EXPECT_OK(output_handle->Close());

  // Check the writen bytes.
  auto input_handle = file::OpenInputFile(file_path).value();
  auto read_content = input_handle->ReadAll().value();
  EXPECT_OK(input_handle->Close());
  EXPECT_EQ(
      "abc,123\n\n,,\nhello "
      "world\n\"a\"\"b\"\"c,xyz\"\n\"a\nb\nc,xyz\"\n",
      read_content);

  // Read back the content with the csv reader.
  input_handle = file::OpenInputFile(file_path).value();
  Reader reader(input_handle.get());
  std::vector<absl::string_view>* row;
  int row_idx = 0;
  while (reader.NextRow(&row).value()) {
    LOG(INFO) << "row_idx:" << row_idx;
    EXPECT_EQ(*row, content[row_idx]);
    row_idx++;
  }
  EXPECT_EQ(row_idx, content.size());
  EXPECT_OK(input_handle->Close());
}

}  // namespace
}  // namespace csv
}  // namespace utils
}  // namespace yggdrasil_decision_forests
