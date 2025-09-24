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

#include "yggdrasil_decision_forests/serving/embed/java/model_data_bank.h"

#include <cstdint>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"
#include "yggdrasil_decision_forests/serving/embed/java/java_embed.h"
#include "yggdrasil_decision_forests/utils/test.h"  // IWYU pragma: keep
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

class ModelDataBankTest : public ::testing::Test {
 protected:
  void SetUp() override {
    stats_.has_conditions
        [model::decision_tree::proto::Condition::kHigherCondition] = true;
    stats_.has_conditions
        [model::decision_tree::proto::Condition::kContainsCondition] = true;
    stats_.has_conditions
        [model::decision_tree::proto::Condition::kObliqueCondition] = true;

    internal_options_.node_offset_bytes = 4;
    internal_options_.feature_index_bytes = 2;
    internal_options_.categorical_idx_bytes = 2;
    internal_options_.numerical_feature_is_float = false;
    internal_options_.feature_value_bytes = 4;

    specialized_conversion_.leaf_value_spec.dtype = proto::DType::FLOAT32;
  }

  ModelStatistics stats_;
  JavaInternalOptions internal_options_;
  SpecializedConversion specialized_conversion_;
};

TEST_F(ModelDataBankTest, SerializeData) {
  ModelDataBank bank(internal_options_, stats_, specialized_conversion_);
  ASSERT_OK(bank.AddNode({.pos = 1,
                          .val = 2.0f,
                          .feat = 3,
                          .thr = static_cast<int64_t>(4),
                          .cat = 5,
                          .obl = 0}));
  ASSERT_OK(bank.AddNode({.pos = 10,
                          .val = 20.0f,
                          .feat = 30,
                          .thr = static_cast<int64_t>(40),
                          .cat = 50,
                          .obl = 2,
                          .oblique_weights = {1.0, 2.0},
                          .oblique_features = {100, 200}}));
  bank.categorical = {true, false, true};
  ASSERT_OK(bank.AddRootDelta(10));
  ASSERT_OK(bank.AddRootDelta(20));
  ASSERT_OK(bank.FinalizeJavaTypes());

  ASSERT_OK_AND_ASSIGN(const std::string serialized_data,
                       bank.SerializeData(internal_options_));

  std::string expected_data;
  // node_pos: size=2, val={1, 10} (int)
  expected_data += std::string("\0\0\0\2", 4);
  expected_data += std::string("\0\0\0\1", 4);
  expected_data += std::string("\0\0\0\12", 4);
  // node_val: size=2, val={2.0f, 20.0f} (float)
  expected_data += std::string("\0\0\0\2", 4);
  expected_data += std::string("@\0\0\0", 4);
  expected_data += std::string("A\xa0\0\0", 4);
  // node_feat: size=2, val={3, 30} (short)
  expected_data += std::string("\0\0\0\2", 4);
  expected_data += std::string("\0\3", 2);
  expected_data += std::string("\0\36", 2);
  // node_thr: size=2, val={4, 40} (int)
  expected_data += std::string("\0\0\0\2", 4);
  expected_data += std::string("\0\0\0\4", 4);
  expected_data += std::string("\0\0\0\50", 4);
  // node_cat: size=2, val={5, 50} (short)
  expected_data += std::string("\0\0\0\2", 4);
  expected_data += std::string("\0\5", 2);
  expected_data += std::string("\0\62", 2);
  // node_obl: size=2, val={0, 2} (byte)
  expected_data += std::string("\0\0\0\2", 4);
  expected_data += std::string("\0", 1);
  expected_data += std::string("\2", 1);
  // root_deltas: size=2, val={10, 20} (int)
  expected_data += std::string("\0\0\0\2", 4);
  expected_data += std::string("\0\0\0\12", 4);
  expected_data += std::string("\0\0\0\24", 4);
  // oblique_weights: size=2, val={1.0, 2.0} (float)
  expected_data += std::string("\0\0\0\2", 4);
  expected_data += std::string("?\x80\0\0", 4);
  expected_data += std::string("@\0\0\0", 4);
  // oblique_features: size=2, val={100, 200} (short)
  expected_data += std::string("\0\0\0\2", 4);
  expected_data += std::string("\0\144", 2);
  expected_data += std::string("\0\310", 2);
  // categorical bank: num_longs=1, val=5
  expected_data += std::string("\0\0\0\1", 4);
  expected_data += std::string("\0\0\0\0\0\0\0\5", 8);

  EXPECT_EQ(serialized_data, expected_data);
}

TEST_F(ModelDataBankTest, SerializeDataWithSentinels) {
  ModelStatistics stats_without_oblique;

  stats_without_oblique.has_conditions
      [model::decision_tree::proto::Condition::kHigherCondition] = true;
  stats_without_oblique.has_conditions
      [model::decision_tree::proto::Condition::kContainsCondition] = true;
  ModelDataBank bank(internal_options_, stats_without_oblique,
                     specialized_conversion_);
  ASSERT_OK(bank.AddNode({.pos = 1,
                          .val = 2.0f,
                          .feat = 3,
                          .thr = static_cast<int64_t>(4),
                          .cat = 5}));
  // A leaf node. `feat`, `thr`, `cat`, are not set and should get
  // sentinel values.
  ASSERT_OK(bank.AddNode({.val = 123.45f}));
  ASSERT_OK(
      bank.AddNode({.pos = 10, .feat = 30, .thr = static_cast<int64_t>(40)}));
  bank.categorical = {true, false, true};
  ASSERT_OK(bank.AddRootDelta(10));
  ASSERT_OK(bank.AddRootDelta(20));
  ASSERT_OK(bank.FinalizeJavaTypes());

  ASSERT_OK_AND_ASSIGN(const std::string serialized_data,
                       bank.SerializeData(internal_options_));

  std::string expected_data;
  // node_pos: size=3, val={1, 0, 10} (int)
  expected_data += std::string("\0\0\0\3", 4);
  expected_data += std::string("\0\0\0\1", 4);
  expected_data += std::string("\0\0\0\0", 4);
  expected_data += std::string("\0\0\0\12", 4);
  // node_val: size=3, val={2.0f, 123.45f, 0.} (float)
  expected_data += std::string("\0\0\0\3", 4);
  expected_data += std::string("@\0\0\0", 4);
  expected_data += std::string("B\xf6\xe6\x66", 4);
  expected_data += std::string("\0\0\0\0", 4);
  // node_feat: size=3, val={3, 0, 30} (short)
  expected_data += std::string("\0\0\0\3", 4);
  expected_data += std::string("\0\3", 2);
  expected_data += std::string("\0\0", 2);
  expected_data += std::string("\0\36", 2);
  // node_thr: size=3, val={4, 0, 40} (int)
  expected_data += std::string("\0\0\0\3", 4);
  expected_data += std::string("\0\0\0\4", 4);
  expected_data += std::string("\0\0\0\0", 4);
  expected_data += std::string("\0\0\0\50", 4);
  // node_cat: size=3, val={5, 0, 50} (short)
  expected_data += std::string("\0\0\0\3", 4);
  expected_data += std::string("\0\5", 2);
  expected_data += std::string("\0\0", 2);
  expected_data += std::string("\0\0", 2);
  // root_deltas, oblique_weights, oblique_features and categorical are not per-
  // node banks and are not affected by missing values in nodes.
  expected_data += std::string("\0\0\0\2", 4);
  expected_data += std::string("\0\0\0\12", 4);
  expected_data += std::string("\0\0\0\24", 4);
  expected_data += std::string("\0\0\0\1", 4);
  expected_data += std::string("\0\0\0\0\0\0\0\5", 8);

  EXPECT_EQ(serialized_data, expected_data);
}

TEST_F(ModelDataBankTest, GenerateJavaCode) {
  ModelDataBank bank(internal_options_, stats_, specialized_conversion_);
  ASSERT_OK(bank.AddNode({.pos = 1,
                          .val = 2.0f,
                          .feat = 3,
                          .thr = static_cast<int64_t>(4),
                          .cat = 5,
                          .obl = 0,
                          .oblique_weights = {1.0f},
                          .oblique_features = {10}}));
  bank.categorical = {true, false, true};
  ASSERT_OK(bank.AddRootDelta(10));
  ASSERT_OK(bank.FinalizeJavaTypes());

  ASSERT_OK_AND_ASSIGN(
      const std::string java_code,
      bank.GenerateJavaCode(internal_options_, "MyModel", "MyModelData.bin"));

  const std::string expected_code =
      R"(  private static final int[] nodePos;
  private static final float[] nodeVal;
  private static final short[] nodeFeat;
  private static final int[] nodeThr;
  private static final short[] nodeCat;
  private static final byte[] nodeObl;
  private static final int[] rootDeltas;
  private static final float[] obliqueWeights;
  private static final short[] obliqueFeatures;
  private static final BitSet categoricalBank;

  static {
    try (InputStream is = MyModel.class.getResourceAsStream("MyModelData.bin");
         DataInputStream dis = new DataInputStream(new BufferedInputStream(is))) {
    int nodePosLength = dis.readInt();
    nodePos = new int[nodePosLength];
    for (int i = 0; i < nodePosLength; i++) {
      nodePos[i] = dis.readInt();
    }
    int nodeValLength = dis.readInt();
    nodeVal = new float[nodeValLength];
    for (int i = 0; i < nodeValLength; i++) {
      nodeVal[i] = dis.readFloat();
    }
    int nodeFeatLength = dis.readInt();
    nodeFeat = new short[nodeFeatLength];
    for (int i = 0; i < nodeFeatLength; i++) {
      nodeFeat[i] = dis.readShort();
    }
    int nodeThrLength = dis.readInt();
    nodeThr = new int[nodeThrLength];
    for (int i = 0; i < nodeThrLength; i++) {
      nodeThr[i] = dis.readInt();
    }
    int nodeCatLength = dis.readInt();
    nodeCat = new short[nodeCatLength];
    for (int i = 0; i < nodeCatLength; i++) {
      nodeCat[i] = dis.readShort();
    }
    int nodeOblLength = dis.readInt();
    nodeObl = new byte[nodeOblLength];
    for (int i = 0; i < nodeOblLength; i++) {
      nodeObl[i] = dis.readByte();
    }
    int rootDeltasLength = dis.readInt();
    rootDeltas = new int[rootDeltasLength];
    for (int i = 0; i < rootDeltasLength; i++) {
      rootDeltas[i] = dis.readInt();
    }
    int obliqueWeightsLength = dis.readInt();
    obliqueWeights = new float[obliqueWeightsLength];
    for (int i = 0; i < obliqueWeightsLength; i++) {
      obliqueWeights[i] = dis.readFloat();
    }
    int obliqueFeaturesLength = dis.readInt();
    obliqueFeatures = new short[obliqueFeaturesLength];
    for (int i = 0; i < obliqueFeaturesLength; i++) {
      obliqueFeatures[i] = dis.readShort();
    }
    int categoricalBankNumLongs = dis.readInt();
    if (categoricalBankNumLongs > 0) {
      long[] longs = new long[categoricalBankNumLongs];
      for (int i = 0; i < categoricalBankNumLongs; i++) {
        longs[i] = dis.readLong();
      }
      categoricalBank = BitSet.valueOf(longs);
    } else {
      categoricalBank = new BitSet();
    }
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model data resource: " + e.getMessage(), e);
    }
  }
)";
  EXPECT_EQ(java_code, expected_code);
}

namespace {}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed::internal
