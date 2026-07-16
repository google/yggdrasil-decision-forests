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

#include "yggdrasil_decision_forests/model/postprocessor/abstract_postprocessor.h"

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/postprocessor/abstract_postprocessor.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"

namespace yggdrasil_decision_forests::model::postprocessor {
namespace {

class FakePostprocessor : public AbstractPostprocessor {
 public:
  FakePostprocessor() = default;

  void Reset() {
    process_dataset_called_ = false;
    process_example_called_ = false;
    export_called_ = false;
  }

  bool process_dataset_called() const { return process_dataset_called_; }
  bool process_example_called() const { return process_example_called_; }
  bool export_called() const { return export_called_; }

 protected:
  void ProcessImpl(const dataset::VerticalDataset& dataset,
                   dataset::VerticalDataset::row_t row_idx,
                   yggdrasil_decision_forests::model::proto::Prediction*
                       prediction) const override {
    process_dataset_called_ = true;
  }

  void ProcessImpl(const dataset::proto::Example& example,
                   yggdrasil_decision_forests::model::proto::Prediction*
                       prediction) const override {
    process_example_called_ = true;
  }

  void ExportProtoImpl(proto::AbstractPostprocessor* proto) const override {
    export_called_ = true;
  }

 private:
  mutable bool process_dataset_called_ = false;
  mutable bool process_example_called_ = false;
  mutable bool export_called_ = false;
};

TEST(AbstractPostprocessorTest, EnableDisable) {
  FakePostprocessor postprocessor;
  EXPECT_TRUE(postprocessor.enabled());

  postprocessor.disable();
  EXPECT_FALSE(postprocessor.enabled());

  postprocessor.enable();
  EXPECT_TRUE(postprocessor.enabled());
}

TEST(AbstractPostprocessorTest, ProcessDatasetWhenEnabled) {
  FakePostprocessor postprocessor;
  dataset::VerticalDataset dataset;
  yggdrasil_decision_forests::model::proto::Prediction prediction;

  postprocessor.Process(dataset, 0, &prediction);
  EXPECT_TRUE(postprocessor.process_dataset_called());
}

TEST(AbstractPostprocessorTest, ProcessDatasetWhenDisabled) {
  FakePostprocessor postprocessor;
  postprocessor.disable();
  dataset::VerticalDataset dataset;
  yggdrasil_decision_forests::model::proto::Prediction prediction;

  postprocessor.Process(dataset, 0, &prediction);
  EXPECT_FALSE(postprocessor.process_dataset_called());
}

TEST(AbstractPostprocessorTest, ProcessExampleWhenEnabled) {
  FakePostprocessor postprocessor;
  dataset::proto::Example example;
  yggdrasil_decision_forests::model::proto::Prediction prediction;

  postprocessor.Process(example, &prediction);
  EXPECT_TRUE(postprocessor.process_example_called());
}

TEST(AbstractPostprocessorTest, ProcessExampleWhenDisabled) {
  FakePostprocessor postprocessor;
  postprocessor.disable();
  dataset::proto::Example example;
  yggdrasil_decision_forests::model::proto::Prediction prediction;

  postprocessor.Process(example, &prediction);
  EXPECT_FALSE(postprocessor.process_example_called());
}

TEST(AbstractPostprocessorTest, ExportProto) {
  FakePostprocessor postprocessor;
  proto::AbstractPostprocessor proto;

  postprocessor.ExportProto(&proto);
  EXPECT_TRUE(proto.enabled());
  EXPECT_TRUE(postprocessor.export_called());

  postprocessor.disable();
  postprocessor.Reset();
  postprocessor.ExportProto(&proto);
  EXPECT_FALSE(proto.enabled());
  EXPECT_TRUE(postprocessor.export_called());
}

}  // namespace
}  // namespace yggdrasil_decision_forests::model::postprocessor
