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

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::dataset::avro {
namespace {

using test::EqualsProto;

std::string DatasetDir() {
  return file::JoinPath(
      test::DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/dataset");
}

TEST(AvroExample, CreateDataspec) {
  dataset::proto::DataSpecificationGuide guide;
  {
    auto* col = guide.add_column_guides();
    col->set_column_name_pattern("^f_another_array_of_string$");
    col->set_type(proto::ColumnType::CATEGORICAL_SET);
  }

  {
    auto* col = guide.add_column_guides();
    col->set_column_name_pattern("^f_another_float$");
    col->set_ignore_column(true);
  }

  guide.mutable_default_column_guide()
      ->mutable_categorial()
      ->set_min_vocab_frequency(1);

  ASSERT_OK_AND_ASSIGN(
      const auto dataspec,
      CreateDataspec(file::JoinPath(DatasetDir(), "toy_codex-null.avro"),
                     guide));
  LOG(INFO) << "Dataspec:\n" << dataset::PrintHumanReadable(dataspec);

  const dataset::proto::DataSpecification expected = PARSE_TEST_PROTO(R"pb(
    columns {
      type: BOOLEAN
      name: "f_boolean"
      boolean { count_true: 1 count_false: 1 }
    }
    columns {
      type: NUMERICAL
      name: "f_int"
      numerical { mean: 5.5 min_value: 5 max_value: 6 standard_deviation: 0.5 }
    }
    columns {
      type: NUMERICAL
      name: "f_long"
      numerical {
        mean: 617222
        min_value: -123
        max_value: 1234567
        standard_deviation: 617345.00049850566
      }
    }
    columns {
      type: NUMERICAL
      name: "f_float"
      numerical {
        mean: 0.95375001430511475
        min_value: -1.234
        max_value: 3.1415
        standard_deviation: 2.1877499915544623
      }
    }
    columns {
      type: NUMERICAL
      name: "f_double"
      numerical {
        mean: 6.7890000343322754
        min_value: 6.789
        max_value: 6.789
        standard_deviation: 0.0011401533426769351
      }
      count_nas: 1
    }
    columns {
      type: CATEGORICAL
      name: "f_string"
      categorical {
        most_frequent_value: 1
        number_of_unique_values: 3
        min_value_count: 1
        max_number_of_unique_values: 2000
        is_already_integerized: false
        items {
          key: ""
          value { index: 2 count: 1 }
        }
        items {
          key: "<OOD>"
          value { index: 0 count: 0 }
        }
        items {
          key: "hello"
          value { index: 1 count: 1 }
        }
      }
    }
    columns {
      type: CATEGORICAL
      name: "f_bytes"
      categorical {
        most_frequent_value: 1
        number_of_unique_values: 3
        min_value_count: 1
        max_number_of_unique_values: 2000
        is_already_integerized: false
        items {
          key: ""
          value { index: 2 count: 1 }
        }
        items {
          key: "<OOD>"
          value { index: 0 count: 0 }
        }
        items {
          key: "world"
          value { index: 1 count: 1 }
        }
      }
    }
    columns {
      type: NUMERICAL
      name: "f_float_optional"
      numerical {
        mean: 6.0999999046325684
        min_value: 6.1
        max_value: 6.1
        standard_deviation: 0.00049795111524192609
      }
      count_nas: 1
    }
    columns {
      type: CATEGORICAL_SET
      name: "f_another_array_of_string"
      is_manual_type: true
      categorical {
        most_frequent_value: 1
        number_of_unique_values: 5
        min_value_count: 1
        max_number_of_unique_values: 2000
        is_already_integerized: false
        items {
          key: "<OOD>"
          value { index: 0 count: 0 }
        }
        items {
          key: "a"
          value { index: 4 count: 1 }
        }
        items {
          key: "b"
          value { index: 3 count: 1 }
        }
        items {
          key: "c"
          value { index: 1 count: 2 }
        }
        items {
          key: "def"
          value { index: 2 count: 1 }
        }
      }
    }
    columns {
      type: NUMERICAL
      name: "f_array_of_float.0_of_3"
      numerical { mean: 2.5 min_value: 1 max_value: 4 standard_deviation: 1.5 }
      count_nas: 0
    }
    columns {
      type: NUMERICAL
      name: "f_array_of_float.1_of_3"
      numerical { mean: 3.5 min_value: 2 max_value: 5 standard_deviation: 1.5 }
      count_nas: 0
    }
    columns {
      type: NUMERICAL
      name: "f_array_of_float.2_of_3"
      numerical { mean: 4.5 min_value: 3 max_value: 6 standard_deviation: 1.5 }
      count_nas: 0
    }
    columns {
      type: NUMERICAL
      name: "f_array_of_double.0_of_3"
      numerical { mean: 25 min_value: 10 max_value: 40 standard_deviation: 15 }
      count_nas: 0
    }
    columns {
      type: NUMERICAL
      name: "f_array_of_double.1_of_3"
      numerical { mean: 35 min_value: 20 max_value: 50 standard_deviation: 15 }
      count_nas: 0
    }
    columns {
      type: NUMERICAL
      name: "f_array_of_double.2_of_3"
      numerical { mean: 45 min_value: 30 max_value: 60 standard_deviation: 15 }
      count_nas: 0
    }
    columns {
      type: CATEGORICAL
      name: "f_array_of_string.0_of_3"
      categorical {
        most_frequent_value: 1
        number_of_unique_values: 3
        min_value_count: 1
        max_number_of_unique_values: 2000
        is_already_integerized: false
        items {
          key: "<OOD>"
          value { index: 0 count: 0 }
        }
        items {
          key: "a"
          value { index: 2 count: 1 }
        }
        items {
          key: "c"
          value { index: 1 count: 1 }
        }
      }
      count_nas: 0
    }
    columns {
      type: CATEGORICAL
      name: "f_array_of_string.1_of_3"
      categorical {
        most_frequent_value: 1
        number_of_unique_values: 3
        min_value_count: 1
        max_number_of_unique_values: 2000
        is_already_integerized: false
        items {
          key: "<OOD>"
          value { index: 0 count: 0 }
        }
        items {
          key: "a"
          value { index: 2 count: 1 }
        }
        items {
          key: "b"
          value { index: 1 count: 1 }
        }
      }
      count_nas: 0
    }
    columns {
      type: CATEGORICAL
      name: "f_array_of_string.2_of_3"
      categorical {
        most_frequent_value: 1
        number_of_unique_values: 3
        min_value_count: 1
        max_number_of_unique_values: 2000
        is_already_integerized: false
        items {
          key: "<OOD>"
          value { index: 0 count: 0 }
        }
        items {
          key: "b"
          value { index: 2 count: 1 }
        }
        items {
          key: "c"
          value { index: 1 count: 1 }
        }
      }
      count_nas: 0
    }
    columns {
      type: NUMERICAL
      name: "f_optional_array_of_float.0_of_3"
      numerical { mean: 0.5 min_value: 1 max_value: 1 standard_deviation: 0.5 }
      count_nas: 0
    }
    columns {
      type: NUMERICAL
      name: "f_optional_array_of_float.1_of_3"
      numerical { mean: 1 min_value: 2 max_value: 2 standard_deviation: 1 }
      count_nas: 0
    }
    columns {
      type: NUMERICAL
      name: "f_optional_array_of_float.2_of_3"
      numerical { mean: 1.5 min_value: 3 max_value: 3 standard_deviation: 1.5 }
      count_nas: 0
    }
    created_num_rows: 2
    unstackeds {
      original_name: "f_array_of_float"
      begin_column_idx: 9
      size: 3
      type: NUMERICAL
    }
    unstackeds {
      original_name: "f_array_of_double"
      begin_column_idx: 12
      size: 3
      type: NUMERICAL
    }
    unstackeds {
      original_name: "f_array_of_string"
      begin_column_idx: 15
      size: 3
      type: CATEGORICAL
    }
    unstackeds {
      original_name: "f_optional_array_of_float"
      begin_column_idx: 18
      size: 3
      type: NUMERICAL
    }
  )pb");

  EXPECT_THAT(dataspec, EqualsProto(expected));
}

struct ReadExampleCase {
  std::string filename;
};

SIMPLE_PARAMETERIZED_TEST(ReadExample, ReadExampleCase,
                          {
                              {"toy_codex-null.avro"},
                              {"toy_codex-deflate.avro"},
                          }) {
  const auto& test_case = GetParam();
  dataset::proto::DataSpecificationGuide guide;
  {
    auto* col = guide.add_column_guides();
    col->set_column_name_pattern("^f_another_array_of_string$");
    col->set_type(proto::ColumnType::CATEGORICAL_SET);
  }

  {
    auto* col = guide.add_column_guides();
    col->set_column_name_pattern("^f_another_float$");
    col->set_ignore_column(true);
  }

  guide.mutable_default_column_guide()
      ->mutable_categorial()
      ->set_min_vocab_frequency(1);

  ASSERT_OK_AND_ASSIGN(
      const auto dataspec,
      CreateDataspec(file::JoinPath(DatasetDir(), test_case.filename), guide));

  AvroExampleReader reader(dataspec, {});
  ASSERT_OK(reader.Open(file::JoinPath(DatasetDir(), test_case.filename)));
  proto::Example example;
  ASSERT_OK_AND_ASSIGN(bool has_next, reader.Next(&example));
  ASSERT_TRUE(has_next);

  const proto::Example expected_1 = PARSE_TEST_PROTO(R"pb(
    attributes { boolean: true }
    attributes { numerical: 5 }
    attributes { numerical: 1234567 }
    attributes { numerical: 3.1415 }
    attributes {}
    attributes { categorical: 1 }
    attributes { categorical: 1 }
    attributes { numerical: 6.1 }
    attributes { categorical_set { values: 4 values: 3 values: 1 } }
    attributes { numerical: 1 }
    attributes { numerical: 2 }
    attributes { numerical: 3 }
    attributes { numerical: 10 }
    attributes { numerical: 20 }
    attributes { numerical: 30 }
    attributes { categorical: 2 }
    attributes { categorical: 1 }
    attributes { categorical: 1 }
    attributes { numerical: 1 }
    attributes { numerical: 2 }
    attributes { numerical: 3 }
  )pb");
  EXPECT_THAT(example, EqualsProto(expected_1));

  ASSERT_OK_AND_ASSIGN(has_next, reader.Next(&example));
  ASSERT_TRUE(has_next);
  const proto::Example expected_2 = PARSE_TEST_PROTO(R"pb(
    attributes { boolean: false }
    attributes { numerical: 6 }
    attributes { numerical: -123 }
    attributes { numerical: -1.234 }
    attributes { numerical: 6.789 }
    attributes { categorical: 2 }
    attributes { categorical: 2 }
    attributes {}
    attributes { categorical_set { values: 1 values: 2 } }
    attributes { numerical: 4 }
    attributes { numerical: 5 }
    attributes { numerical: 6 }
    attributes { numerical: 40 }
    attributes { numerical: 50 }
    attributes { numerical: 60 }
    attributes { categorical: 1 }
    attributes { categorical: 2 }
    attributes { categorical: 2 }
    attributes {}
    attributes {}
    attributes {}
  )pb");
  EXPECT_THAT(example, EqualsProto(expected_2));

  ASSERT_OK_AND_ASSIGN(has_next, reader.Next(&example));
  ASSERT_FALSE(has_next);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::dataset::avro
