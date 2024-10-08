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

#include "yggdrasil_decision_forests/dataset/avro_example.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::dataset {
namespace {

std::string DatasetDir() {
  return file::JoinPath(
      test::DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/dataset");
}

struct ReadIntegerCase {
  std::string input;
  int64_t expected_value;
};

SIMPLE_PARAMETERIZED_TEST(ReadInteger, ReadIntegerCase,
                          {
                              {std::string("\x00", 1), 0},
                              {"\x01", -1},
                              {"\x02", 1},
                              {"\x03", -2},
                              {"\x04", 2},
                              {"\x05", -3},
                              {"\x06", 3},
                              {"\x07", -4},
                              {"\x7F", -64},
                              {"\x80\x01", 64},
                              {"\x80\x02", 128},
                              {"\x82\x01", 65},
                              {"\x80\x03", 192},
                              {"\x80\x04", 256},
                              {"\xFE\x01", 127},
                              {"\x80\x80\x01", 1 << (7 * 2 - 1)},
                              {"\x80\x80\x80\x01", 1 << (7 * 3 - 1)},
                              {"\x80\x80\x80\x80\x80\x80\x80\x01",
                               static_cast<int64_t>(1) << (7 * 7 - 1)},
                          }) {
  const auto& test_case = GetParam();
  utils::StringInputByteStream stream(test_case.input);
  ASSERT_OK_AND_ASSIGN(const auto value, internal::ReadInteger(&stream));
  EXPECT_EQ(value, test_case.expected_value);
}

struct ReadFloatCase {
  std::string input;
  float expected_value;
};

SIMPLE_PARAMETERIZED_TEST(ReadFloat, ReadFloatCase,
                          {
                              {std::string("\x00\x00\x00\x00", 4), 0.f},
                              {std::string("\x00\x00\x00\x40", 4), 2.f},
                              {std::string("\x00\x00\x40\x40", 4), 3.f},
                              {std::string("\x00\x00\x60\x40", 4), 3.5f},
                              {std::string("\x00\x00\x80\x7F", 4),
                               std::numeric_limits<float>::infinity()},
                              {std::string("\x00\x00\x80\xFF", 4),
                               -std::numeric_limits<float>::infinity()},
                          }) {
  const auto& test_case = GetParam();
  utils::StringInputByteStream stream(test_case.input);
  ASSERT_OK_AND_ASSIGN(const auto value, internal::ReadFloat(&stream));
  EXPECT_EQ(value, test_case.expected_value);
}

TEST(ReadFloat, IsNan) {
  utils::StringInputByteStream stream("\xFF\xFF\xFF\x7F");
  ASSERT_OK_AND_ASSIGN(const auto value, internal::ReadFloat(&stream));
  EXPECT_TRUE(std::isnan(value));
}

struct ReadDoubleCase {
  std::string input;
  double expected_value;
};

SIMPLE_PARAMETERIZED_TEST(
    ReadDouble, ReadDoubleCase,
    {
        {std::string("\x00\x00\x00\x00\x00\x00\x00\x00", 8), 0.},
        {std::string("\x00\x00\x00\x00\x00\x00\x00\x40", 8), 2.},
        {std::string("\x00\x00\x00\x00\x00\x00\x40\x40", 8), 32.},
        {std::string("\x00\x00\x00\x00\x00\x00\x60\x40", 8), 128.},
    }) {
  const auto& test_case = GetParam();
  utils::StringInputByteStream stream(test_case.input);
  ASSERT_OK_AND_ASSIGN(const auto value, internal::ReadDouble(&stream));
  EXPECT_EQ(value, test_case.expected_value);
}

struct ReadBooleanCase {
  std::string input;
  bool expected_value;
};

SIMPLE_PARAMETERIZED_TEST(ReadBoolean, ReadBooleanCase,
                          {
                              {std::string("\x00", 1), false},
                              {std::string("\x01", 1), true},
                          }) {
  const auto& test_case = GetParam();
  utils::StringInputByteStream stream(test_case.input);
  ASSERT_OK_AND_ASSIGN(const auto value, internal::ReadBoolean(&stream));
  EXPECT_EQ(value, test_case.expected_value);
}

struct ReadStringCase {
  std::string input;
  std::string expected_value;
};

SIMPLE_PARAMETERIZED_TEST(ReadString, ReadStringCase,
                          {
                              {std::string("\x00", 1), ""},
                              {"\x02\x41", "A"},
                              {"\x16Hello World", "Hello World"},
                              {"\x06\x66\x6F\x6F", "foo"},
                          }) {
  const auto& test_case = GetParam();
  utils::StringInputByteStream stream(test_case.input);
  ASSERT_OK_AND_ASSIGN(const auto value, internal::ReadString(&stream));
  EXPECT_EQ(value, test_case.expected_value);
}

TEST(AvroExample, ExtractSchema) {
  ASSERT_OK_AND_ASSIGN(const auto schema,
                       AvroReader::ExtractSchema(file::JoinPath(
                           DatasetDir(), "toy_codex-null.avro")));
  EXPECT_EQ(
      schema,
      R"({"type": "record", "name": "ToyDataset", "fields": [{"name": "f_null", "type": "null"}, {"name": "f_boolean", "type": "boolean"}, {"name": "f_int", "type": "int"}, {"name": "f_long", "type": "long"}, {"name": "f_float", "type": "float"}, {"name": "f_another_float", "type": "float"}, {"name": "f_double", "type": "double"}, {"name": "f_string", "type": "string"}, {"name": "f_bytes", "type": "bytes"}, {"name": "f_float_optional", "type": ["null", "float"]}, {"name": "f_array_of_float", "type": {"type": "array", "items": "float"}}, {"name": "f_array_of_double", "type": {"type": "array", "items": "double"}}, {"name": "f_array_of_string", "type": {"type": "array", "items": "string"}}, {"name": "f_another_array_of_string", "type": {"type": "array", "items": "string"}}, {"name": "f_optional_array_of_float", "type": ["null", {"type": "array", "items": "float"}]}], "__fastavro_parsed": true})");
}

TEST(AvroExample, ExtractSchemaCompressed) {
  ASSERT_OK_AND_ASSIGN(const auto schema,
                       AvroReader::ExtractSchema(file::JoinPath(
                           DatasetDir(), "toy_codex-deflate.avro")));
  EXPECT_EQ(
      schema,
      R"({"type": "record", "name": "ToyDataset", "fields": [{"name": "f_null", "type": "null"}, {"name": "f_boolean", "type": "boolean"}, {"name": "f_int", "type": "int"}, {"name": "f_long", "type": "long"}, {"name": "f_float", "type": "float"}, {"name": "f_another_float", "type": "float"}, {"name": "f_double", "type": "double"}, {"name": "f_string", "type": "string"}, {"name": "f_bytes", "type": "bytes"}, {"name": "f_float_optional", "type": ["null", "float"]}, {"name": "f_array_of_float", "type": {"type": "array", "items": "float"}}, {"name": "f_array_of_double", "type": {"type": "array", "items": "double"}}, {"name": "f_array_of_string", "type": {"type": "array", "items": "string"}}, {"name": "f_another_array_of_string", "type": {"type": "array", "items": "string"}}, {"name": "f_optional_array_of_float", "type": ["null", {"type": "array", "items": "float"}]}], "__fastavro_parsed": true})");
}

}  // namespace
}  // namespace yggdrasil_decision_forests::dataset
