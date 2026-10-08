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

// Test the predictions value of embedded models.
//
// Golden model predictions are generated with the following pattern in python:
/*

import pandas as pd
import ydf

ydf_root = ...
model_path = f"{ydf_root}/test_data/model/adult_binary_class_rf"
ds_path = f"{ydf_root}/test_data/dataset/adult_test.csv"
model = ydf.load_model(model_path)
ds = pd.read_csv(ds_path)
print(ds.iloc[0])
print(model.predict(ds[:1]))
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_abalone_regression_gbdt_v2.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_abalone_regression_gbdt_v2_routing.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_abalone_regression_rf_small.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_abalone_regression_rf_small_routing.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_gbdt_calibrated.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_gbdt_calibrated_routing.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_gbdt_filegroup_filegroup.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_gbdt_integerized_proba.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_gbdt_integerized_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_gbdt_oblique_proba.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_gbdt_v2_class.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_gbdt_v2_proba.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_gbdt_v2_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_gbdt_v2_proba_routing_with_string_vocab.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_gbdt_v2_score.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_rf_nwta_small_class.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_rf_nwta_small_proba.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_rf_nwta_small_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_rf_nwta_small_score.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_rf_wta_small_class.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_rf_wta_small_proba.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_rf_wta_small_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_adult_binary_class_rf_wta_small_score.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_gbdt_v2_class.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_gbdt_v2_proba.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_gbdt_v2_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_gbdt_v2_score.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_rf_nwta_small_class.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_rf_nwta_small_proba.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_rf_nwta_small_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_rf_nwta_small_score.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_rf_wta_small_class.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_rf_wta_small_proba.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_rf_wta_small_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/test_model_iris_multi_class_rf_wta_small_score.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::serving::embed {
namespace {

constexpr double eps = 0.00003;
constexpr int kMaxNumTrees = 200;

template <typename T = float>
T Num(const dataset::VerticalDataset& ds, int row, absl::string_view name) {
  const int col = dataset::GetColumnIdxFromName(name, ds.data_spec());
  return ds.ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(col)
      ->values()[row];
}

template <typename Enum>
Enum Cat(const dataset::VerticalDataset& ds, int row, absl::string_view name) {
  const int col = dataset::GetColumnIdxFromName(name, ds.data_spec());
  const auto* c =
      ds.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(col);
  const int32_t val =
      c->IsNa(row)
          ? ds.data_spec().columns(col).categorical().most_frequent_value()
          : c->values()[row];
  return static_cast<Enum>(val);
}

std::string CatStr(const dataset::VerticalDataset& ds, int row,
                   absl::string_view name) {
  const int col = dataset::GetColumnIdxFromName(name, ds.data_spec());
  const auto* c =
      ds.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(col);
  const auto& spec = ds.data_spec().columns(col);
  const int32_t val = c->IsNa(row) ? spec.categorical().most_frequent_value()
                                   : c->values()[row];
  return dataset::CategoricalIdxToRepresentation(spec, val);
}

int32_t CatInt(const dataset::VerticalDataset& ds, int row,
               absl::string_view name) {
  const int col = dataset::GetColumnIdxFromName(name, ds.data_spec());
  return ds.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(col)
      ->values()[row];
}

#define ADULT_EXAMPLE                                                      \
  [](const dataset::VerticalDataset& ds, int row) {                        \
    return Instance{                                                       \
        .age = Num<int32_t>(ds, row, "age"),                               \
        .workclass = Cat<FeatureWorkclass>(ds, row, "workclass"),          \
        .fnlwgt = Num<int32_t>(ds, row, "fnlwgt"),                         \
        .education = Cat<FeatureEducation>(ds, row, "education"),          \
        .education_num = Num<int32_t>(ds, row, "education_num"),           \
        .marital_status =                                                  \
            Cat<FeatureMaritalStatus>(ds, row, "marital_status"),          \
        .occupation = Cat<FeatureOccupation>(ds, row, "occupation"),       \
        .relationship = Cat<FeatureRelationship>(ds, row, "relationship"), \
        .race = Cat<FeatureRace>(ds, row, "race"),                         \
        .sex = Cat<FeatureSex>(ds, row, "sex"),                            \
        .capital_gain = Num<int32_t>(ds, row, "capital_gain"),             \
        .capital_loss = Num<int32_t>(ds, row, "capital_loss"),             \
        .hours_per_week = Num<int32_t>(ds, row, "hours_per_week"),         \
        .native_country =                                                  \
            Cat<FeatureNativeCountry>(ds, row, "native_country"),          \
    };                                                                     \
  }

#define ADULT_EXAMPLE_STRING_CAT                                               \
  [](const dataset::VerticalDataset& ds, int row) {                            \
    return Instance{                                                           \
        .age = Num<int32_t>(ds, row, "age"),                                   \
        .workclass = FeatureWorkclassFromString(CatStr(ds, row, "workclass")), \
        .fnlwgt = Num<int32_t>(ds, row, "fnlwgt"),                             \
        .education = FeatureEducationFromString(CatStr(ds, row, "education")), \
        .education_num = Num<int32_t>(ds, row, "education_num"),               \
        .marital_status =                                                      \
            FeatureMaritalStatusFromString(CatStr(ds, row, "marital_status")), \
        .occupation =                                                          \
            FeatureOccupationFromString(CatStr(ds, row, "occupation")),        \
        .relationship =                                                        \
            FeatureRelationshipFromString(CatStr(ds, row, "relationship")),    \
        .race = FeatureRaceFromString(CatStr(ds, row, "race")),                \
        .sex = FeatureSexFromString(CatStr(ds, row, "sex")),                   \
        .capital_gain = Num<int32_t>(ds, row, "capital_gain"),                 \
        .capital_loss = Num<int32_t>(ds, row, "capital_loss"),                 \
        .hours_per_week = Num<int32_t>(ds, row, "hours_per_week"),             \
        .native_country =                                                      \
            FeatureNativeCountryFromString(CatStr(ds, row, "native_country")), \
    };                                                                         \
  }

#define ADULT_INTEGERIZED_EXAMPLE                                  \
  [](const dataset::VerticalDataset& ds, int row) {                \
    return Instance{                                               \
        .workclass = CatInt(ds, row, "workclass"),                 \
        .education = CatInt(ds, row, "education"),                 \
        .marital_status = CatInt(ds, row, "marital_status"),       \
        .occupation = CatInt(ds, row, "occupation"),               \
        .relationship = CatInt(ds, row, "relationship"),           \
        .race = CatInt(ds, row, "race"),                           \
        .sex = CatInt(ds, row, "sex"),                             \
        .native_country = CatInt(ds, row, "native_country"),       \
        .age = Num<int32_t>(ds, row, "age"),                       \
        .fnlwgt = Num<int32_t>(ds, row, "fnlwgt"),                 \
        .education_num = Num<int32_t>(ds, row, "education_num"),   \
        .capital_gain = Num<int32_t>(ds, row, "capital_gain"),     \
        .capital_loss = Num<int32_t>(ds, row, "capital_loss"),     \
        .hours_per_week = Num<int32_t>(ds, row, "hours_per_week"), \
    };                                                             \
  }

#define ADULT_INTEGERIZED_EXAMPLE_WITH_NA \
  {                                       \
      .workclass = -1,                    \
      .education = 10,                    \
      .marital_status = 5,                \
      .occupation = 1,                    \
      .relationship = 2,                  \
      .race = 5,                          \
      .sex = 2,                           \
      .native_country = 39,               \
      .age = 39,                          \
      .fnlwgt = 77516,                    \
      .education_num = 13,                \
      .capital_gain = 2174,               \
      .capital_loss = 0,                  \
      .hours_per_week = 40,               \
  }

#define IRIS_EXAMPLE                                  \
  [](const dataset::VerticalDataset& ds, int row) {   \
    return Instance{                                  \
        .sepal_length = Num(ds, row, "Sepal.Length"), \
        .sepal_width = Num(ds, row, "Sepal.Width"),   \
        .petal_length = Num(ds, row, "Petal.Length"), \
        .petal_width = Num(ds, row, "Petal.Width"),   \
    };                                                \
  }

#define ABALONE_EXAMPLE                                 \
  [](const dataset::VerticalDataset& ds, int row) {     \
    return Instance{                                    \
        .type = Cat<FeatureType>(ds, row, "Type"),      \
        .longestshell = Num(ds, row, "LongestShell"),   \
        .diameter = Num(ds, row, "Diameter"),           \
        .height = Num(ds, row, "Height"),               \
        .wholeweight = Num(ds, row, "WholeWeight"),     \
        .shuckedweight = Num(ds, row, "ShuckedWeight"), \
        .visceraweight = Num(ds, row, "VisceraWeight"), \
        .shellweight = Num(ds, row, "ShellWeight"),     \
    };                                                  \
  }

#define ABALONE_EXAMPLE_LITERAL \
  {                             \
      .type = FeatureType::kM,  \
      .longestshell = 0.455f,   \
      .diameter = 0.365f,       \
      .height = 0.095f,         \
      .wholeweight = 0.514f,    \
      .shuckedweight = 0.2245f, \
      .visceraweight = 0.101f,  \
      .shellweight = 0.15f,     \
  }

#define ABALONE_EXAMPLE_WITH_NA \
  {                             \
      .type = FeatureType::kM,  \
      .longestshell = NAN,      \
      .diameter = 0.365f,       \
      .height = NAN,            \
      .wholeweight = 0.514f,    \
      .shuckedweight = NAN,     \
      .visceraweight = NAN,     \
      .shellweight = NAN,       \
  }

template <typename Label, typename = std::enable_if_t<std::is_enum_v<Label>>>
bool ExpectEqualPrediction(const Label pred,
                           const model::proto::Prediction& ydf_pred) {
  EXPECT_EQ(static_cast<int>(pred), ydf_pred.classification().value() - 1);
  return static_cast<int>(pred) == ydf_pred.classification().value() - 1;
}

bool ExpectEqualPrediction(const uint8_t pred,
                           const model::proto::Prediction& ydf_pred) {
  EXPECT_EQ(pred, ydf_pred.classification().distribution().counts(2));
  return pred == ydf_pred.classification().distribution().counts(2);
}

bool ExpectEqualPrediction(const float pred,
                           const model::proto::Prediction& ydf_pred) {
  if (ydf_pred.has_regression()) {
    EXPECT_NEAR(pred, ydf_pred.regression().value(), eps);
    return std::abs(pred - ydf_pred.regression().value()) < eps;
  } else if (ydf_pred.classification().has_logits()) {
    EXPECT_NEAR(pred, ydf_pred.classification().logits().counts(2), eps);
    return std::abs(pred - ydf_pred.classification().logits().counts(2)) < eps;
  } else {
    const auto& dist = ydf_pred.classification().distribution();
    EXPECT_NEAR(pred, dist.counts(2) / dist.sum(), eps);
    return std::abs(pred - dist.counts(2) / dist.sum()) < eps;
  }
}

bool ExpectEqualPrediction(const std::array<uint8_t, 3>& pred,
                           const model::proto::Prediction& ydf_pred) {
  bool ret = true;
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(pred[i], ydf_pred.classification().distribution().counts(i + 1));
    ret &= pred[i] == ydf_pred.classification().distribution().counts(i + 1);
  }
  return ret;
}

bool ExpectEqualPrediction(const std::array<float, 3>& pred,
                           const model::proto::Prediction& ydf_pred) {
  const auto& c = ydf_pred.classification();
  bool ret = true;
  for (int i = 0; i < 3; ++i) {
    const float expected = c.has_logits() ? c.logits().counts(i + 1)
                                          : c.distribution().counts(i + 1) /
                                                c.distribution().sum();
    EXPECT_NEAR(pred[i], expected, eps);
    ret &= std::abs(pred[i] - expected) < eps;
  }
  return ret;
}

template <typename MakeInstanceFn, typename PredictFn>
void CheckPredictions(absl::string_view model_name,
                      absl::string_view dataset_name,
                      MakeInstanceFn make_instance, PredictFn predict,
                      bool output_logits = false) {
  const std::string test_data_dir =
      file::JoinPath(test::DataRootDirectory(),
                     "yggdrasil_decision_forests/test_data");
  ASSERT_OK_AND_ASSIGN(auto model, model::LoadModel(file::JoinPath(
                                       test_data_dir, "model", model_name)));
  if (output_logits) {
    if (auto* gbt = dynamic_cast<
            model::gradient_boosted_trees::GradientBoostedTreesModel*>(
            model.get())) {
      gbt->set_output_logits(true);
    }
  }

  const std::string ds_path = absl::StrCat(
      "csv:", file::JoinPath(test_data_dir, "dataset", dataset_name));
  dataset::VerticalDataset dataset;
  if (model_name == "adult_binary_class_gbdt_integerized") {
    ASSERT_OK_AND_ASSIGN(
        auto ref_model,
        model::LoadModel(
            file::JoinPath(test_data_dir, "model/adult_binary_class_gbdt_v2")));
    dataset::VerticalDataset ref_dataset;
    ASSERT_OK(dataset::LoadVerticalDataset(ds_path, ref_model->data_spec(),
                                           &ref_dataset));
    dataset.set_data_spec(model->data_spec());
    ASSERT_OK(dataset.CreateColumnsFromDataspec());
    dataset.Resize(ref_dataset.nrow());
    for (int col = 0; col < model->data_spec().columns_size(); ++col) {
      const int ref_col = dataset::GetColumnIdxFromName(
          model->data_spec().columns(col).name(), ref_dataset.data_spec());
      if (model->data_spec().columns(col).type() ==
          dataset::proto::ColumnType::NUMERICAL) {
        *dataset
             .MutableColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                 col)
             ->mutable_values() =
            ref_dataset
                .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                    ref_col)
                ->values();
      } else {
        *dataset
             .MutableColumnWithCast<
                 dataset::VerticalDataset::CategoricalColumn>(col)
             ->mutable_values() =
            ref_dataset
                .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                    ref_col)
                ->values();
      }
    }
  } else {
    ASSERT_OK(
        dataset::LoadVerticalDataset(ds_path, model->data_spec(), &dataset));
  }

  const int num_rows = std::min<int>(dataset.nrow(), kMaxNumTrees);
  for (int row = 0; row < num_rows; ++row) {
    model::proto::Prediction ydf_pred;
    model->Predict(dataset, row, &ydf_pred);
    bool res =
        ExpectEqualPrediction(predict(make_instance(dataset, row)), ydf_pred);
    if (!res) {
      LOG(INFO) << "Violation in row " << row;
    }
  }
}

TEST(Embed, test_model_adult_binary_class_gbdt_filegroup_filegroup) {
  using namespace test_model_adult_binary_class_gbdt_filegroup_filegroup;
  const Label pred = Predict(Instance{});
  (void)pred;
}

// GBT binary class

TEST(Embed, test_model_adult_binary_class_gbdt_v2_class) {
  using namespace test_model_adult_binary_class_gbdt_v2_class;
  CheckPredictions("adult_binary_class_gbdt_v2", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_proba) {
  using namespace test_model_adult_binary_class_gbdt_v2_proba;
  CheckPredictions("adult_binary_class_gbdt_v2", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_proba_routing) {
  using namespace test_model_adult_binary_class_gbdt_v2_proba_routing;
  CheckPredictions("adult_binary_class_gbdt_v2", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_gbdt_calibrated) {
  using namespace test_model_adult_binary_class_gbdt_calibrated;
  CheckPredictions("adult_binary_class_gbdt_calibrated", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_gbdt_calibrated_routing) {
  using namespace test_model_adult_binary_class_gbdt_calibrated_routing;
  CheckPredictions("adult_binary_class_gbdt_calibrated", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed,
     test_model_adult_binary_class_gbdt_v2_proba_routing_with_string_vocab) {
  using namespace test_model_adult_binary_class_gbdt_v2_proba_routing_with_string_vocab;
  CheckPredictions("adult_binary_class_gbdt_v2", "adult_test.csv",
                   ADULT_EXAMPLE_STRING_CAT, Predict);
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_score) {
  using namespace test_model_adult_binary_class_gbdt_v2_score;
  CheckPredictions("adult_binary_class_gbdt_v2", "adult_test.csv",
                   ADULT_EXAMPLE, Predict, /*output_logits=*/true);
}

TEST(Embed, test_model_adult_binary_class_gbdt_oblique_proba) {
  using namespace test_model_adult_binary_class_gbdt_oblique_proba;
  CheckPredictions("adult_binary_class_gbdt_oblique", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_gbdt_integerized_proba_routing) {
  using namespace test_model_adult_binary_class_gbdt_integerized_proba_routing;
  CheckPredictions("adult_binary_class_gbdt_integerized", "adult_test.csv",
                   ADULT_INTEGERIZED_EXAMPLE, Predict);
  const float pred_with_na = Predict(ADULT_INTEGERIZED_EXAMPLE_WITH_NA);
  EXPECT_NEAR(pred_with_na, 0.08136991, eps);
}

TEST(Embed, test_model_adult_binary_class_gbdt_integerized_proba) {
  using namespace test_model_adult_binary_class_gbdt_integerized_proba;
  CheckPredictions("adult_binary_class_gbdt_integerized", "adult_test.csv",
                   ADULT_INTEGERIZED_EXAMPLE, Predict);
  const float pred_with_na = Predict(ADULT_INTEGERIZED_EXAMPLE_WITH_NA);
  EXPECT_NEAR(pred_with_na, 0.08136991, eps);
}

// RF binary class

TEST(Embed, test_model_adult_binary_class_rf_nwta_small_class) {
  using namespace test_model_adult_binary_class_rf_nwta_small_class;
  CheckPredictions("adult_binary_class_rf_nwta_small", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_rf_nwta_small_proba) {
  using namespace test_model_adult_binary_class_rf_nwta_small_proba;
  CheckPredictions("adult_binary_class_rf_nwta_small", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_rf_nwta_small_proba_routing) {
  using namespace test_model_adult_binary_class_rf_nwta_small_proba_routing;
  CheckPredictions("adult_binary_class_rf_nwta_small", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_rf_nwta_small_score) {
  using namespace test_model_adult_binary_class_rf_nwta_small_score;
  CheckPredictions("adult_binary_class_rf_nwta_small", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_rf_wta_small_class) {
  using namespace test_model_adult_binary_class_rf_wta_small_class;
  CheckPredictions("adult_binary_class_rf_wta_small", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_rf_wta_small_proba) {
  using namespace test_model_adult_binary_class_rf_wta_small_proba;
  CheckPredictions("adult_binary_class_rf_wta_small", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_rf_wta_small_proba_routing) {
  using namespace test_model_adult_binary_class_rf_wta_small_proba_routing;
  CheckPredictions("adult_binary_class_rf_wta_small", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

TEST(Embed, test_model_adult_binary_class_rf_wta_small_score) {
  using namespace test_model_adult_binary_class_rf_wta_small_score;
  CheckPredictions("adult_binary_class_rf_wta_small", "adult_test.csv",
                   ADULT_EXAMPLE, Predict);
}

// Regression

TEST(Embed, test_model_abalone_regression_gbdt_v2) {
  using namespace test_model_abalone_regression_gbdt_v2;
  CheckPredictions("abalone_regression_gbdt_v2", "abalone.csv", ABALONE_EXAMPLE,
                   Predict);
}

TEST(Embed, test_model_abalone_regression_gbdt_v2_routing) {
  using namespace test_model_abalone_regression_gbdt_v2_routing;
  CheckPredictions("abalone_regression_gbdt_v2", "abalone.csv", ABALONE_EXAMPLE,
                   Predict);
}

TEST(Embed, test_model_abalone_regression_rf_small) {
  using namespace test_model_abalone_regression_rf_small;
  CheckPredictions("abalone_regression_rf_small", "abalone.csv",
                   ABALONE_EXAMPLE, Predict);
}

TEST(Embed, test_model_abalone_regression_rf_small_routing) {
  using namespace test_model_abalone_regression_rf_small_routing;
  CheckPredictions("abalone_regression_rf_small", "abalone.csv",
                   ABALONE_EXAMPLE, Predict);
}

// GBT multi-class

TEST(Embed, test_model_iris_multi_class_gbdt_v2_class) {
  using namespace test_model_iris_multi_class_gbdt_v2_class;
  CheckPredictions("iris_multi_class_gbdt_v2", "iris.csv", IRIS_EXAMPLE,
                   Predict);
}

TEST(Embed, test_model_iris_multi_class_gbdt_v2_score) {
  using namespace test_model_iris_multi_class_gbdt_v2_score;
  CheckPredictions("iris_multi_class_gbdt_v2", "iris.csv", IRIS_EXAMPLE,
                   Predict, /*output_logits=*/true);
}

TEST(Embed, test_model_iris_multi_class_gbdt_v2_proba) {
  using namespace test_model_iris_multi_class_gbdt_v2_proba;
  CheckPredictions("iris_multi_class_gbdt_v2", "iris.csv", IRIS_EXAMPLE,
                   Predict);
}

TEST(Embed, test_model_iris_multi_class_gbdt_v2_proba_routing) {
  using namespace test_model_iris_multi_class_gbdt_v2_proba_routing;
  CheckPredictions("iris_multi_class_gbdt_v2", "iris.csv", IRIS_EXAMPLE,
                   Predict);
}

// RF multi-class

TEST(Embed, test_model_iris_multi_class_rf_nwta_small_class) {
  using namespace test_model_iris_multi_class_rf_nwta_small_class;
  CheckPredictions("iris_multi_class_rf_nwta_small", "iris.csv", IRIS_EXAMPLE,
                   Predict);
}

TEST(Embed, test_model_iris_multi_class_rf_nwta_small_score) {
  using namespace test_model_iris_multi_class_rf_nwta_small_score;
  CheckPredictions("iris_multi_class_rf_nwta_small", "iris.csv", IRIS_EXAMPLE,
                   Predict);
}

TEST(Embed, test_model_iris_multi_class_rf_nwta_small_proba) {
  using namespace test_model_iris_multi_class_rf_nwta_small_proba;
  CheckPredictions("iris_multi_class_rf_nwta_small", "iris.csv", IRIS_EXAMPLE,
                   Predict);
}

TEST(Embed, test_model_iris_multi_class_rf_nwta_small_proba_routing) {
  using namespace test_model_iris_multi_class_rf_nwta_small_proba_routing;
  CheckPredictions("iris_multi_class_rf_nwta_small", "iris.csv", IRIS_EXAMPLE,
                   Predict);
}

TEST(Embed, test_model_iris_multi_class_rf_wta_small_class) {
  using namespace test_model_iris_multi_class_rf_wta_small_class;
  CheckPredictions("iris_multi_class_rf_wta_small", "iris.csv", IRIS_EXAMPLE,
                   Predict);
}

TEST(Embed, test_model_iris_multi_class_rf_wta_small_score) {
  using namespace test_model_iris_multi_class_rf_wta_small_score;
  CheckPredictions("iris_multi_class_rf_wta_small", "iris.csv", IRIS_EXAMPLE,
                   Predict);
}

TEST(Embed, test_model_iris_multi_class_rf_wta_small_proba) {
  using namespace test_model_iris_multi_class_rf_wta_small_proba;
  CheckPredictions("iris_multi_class_rf_wta_small", "iris.csv", IRIS_EXAMPLE,
                   Predict);
}

TEST(Embed, test_model_iris_multi_class_rf_wta_small_proba_routing) {
  using namespace test_model_iris_multi_class_rf_wta_small_proba_routing;
  CheckPredictions("iris_multi_class_rf_wta_small", "iris.csv", IRIS_EXAMPLE,
                   Predict);
}

//
// NA Values

TEST(Embed, test_model_abalone_regression_gbdt_v2_no_na_handling) {
  using namespace test_model_abalone_regression_gbdt_v2;
  const float pred = Predict(ABALONE_EXAMPLE_LITERAL);
  EXPECT_NEAR(pred, 9.815921, eps);
}

TEST(Embed, test_model_abalone_regression_gbdt_v2_routing_no_na_handling) {
  using namespace test_model_abalone_regression_gbdt_v2_routing;
  const float pred = Predict(ABALONE_EXAMPLE_LITERAL);
  EXPECT_NEAR(pred, 9.815921, eps);
}

TEST(Embed, test_model_abalone_regression_gbdt_v2_with_na) {
  using namespace test_model_abalone_regression_gbdt_v2;
  const float pred = Predict(ABALONE_EXAMPLE_WITH_NA);
  EXPECT_NEAR(pred, 9.362932, eps);
}

TEST(Embed, test_model_abalone_regression_gbdt_v2_routing_with_na) {
  using namespace test_model_abalone_regression_gbdt_v2_routing;
  const float pred = Predict(ABALONE_EXAMPLE_WITH_NA);
  EXPECT_NEAR(pred, 9.362932, eps);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed
