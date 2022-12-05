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

#include "yggdrasil_decision_forests/learner/multitasker/multitasker.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/multitasker/multitasker.pb.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/multitasker/multitasker.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace multitasker {
namespace {

class MultitaskerOnAdult : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(MultitaskerLearner::kRegisteredName);
    train_config_.set_label("unused");
    // For the util tester.
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    dataset_filename_ = "adult.csv";
  }
};

TEST_F(MultitaskerOnAdult, Base) {
  auto* mt_config =
      train_config_.MutableExtension(multitasker::proto::multitasker_config);

  mt_config->mutable_base_learner()->set_learner("RANDOM_FOREST");
  generic_parameters_ = model::proto::GenericHyperParameters();
  auto* field = generic_parameters_->add_fields();
  field->set_name("num_trees");
  field->mutable_value()->set_integer(50);

  auto* t1 = mt_config->add_subtasks()->mutable_train_config();
  auto* t2 = mt_config->add_subtasks()->mutable_train_config();
  auto* t3 = mt_config->add_subtasks()->mutable_train_config();

  t1->set_label("income");
  t1->set_task(model::proto::Task::CLASSIFICATION);

  t2->set_label("age");
  t2->set_task(model::proto::Task::REGRESSION);

  t3->set_label("relationship");
  t3->set_task(model::proto::Task::CLASSIFICATION);

  TrainAndEvaluateModel();
  YDF_EXPECT_METRIC_NEAR(metric::Accuracy(evaluation_), 0.860, 0.01);

  utils::RandomEngine rnd(1234);

  auto* mt_model = dynamic_cast<MultitaskerModel*>(model_.get());
  EXPECT_EQ(mt_model->models().size(), 3);

  {
    auto* submodel = mt_model->model(0);
    EXPECT_EQ(submodel->label(), "income");
    metric::proto::EvaluationOptions eval_options;
    eval_options.set_task(model::proto::Task::CLASSIFICATION);
    auto eval = submodel->Evaluate(test_dataset_, eval_options, &rnd);
    YDF_EXPECT_METRIC_NEAR(metric::Accuracy(eval), 0.860, 0.01);
  }

  {
    auto* submodel = mt_model->model(1);
    EXPECT_EQ(submodel->label(), "age");
    metric::proto::EvaluationOptions eval_options;
    eval_options.set_task(model::proto::Task::REGRESSION);
    auto eval = submodel->Evaluate(test_dataset_, eval_options, &rnd);
    YDF_EXPECT_METRIC_NEAR(metric::RMSE(eval), 9.957, 0.05);
  }

  {
    auto* submodel = mt_model->model(2);
    EXPECT_EQ(submodel->label(), "relationship");
    metric::proto::EvaluationOptions eval_options;
    eval_options.set_task(model::proto::Task::CLASSIFICATION);
    auto eval = submodel->Evaluate(test_dataset_, eval_options, &rnd);
    YDF_EXPECT_METRIC_NEAR(metric::Accuracy(eval), 0.786, 0.01);
  }

  {
    // Test serialization with prefix.
    const std::string prefix_model_path =
        file::JoinPath(test::TmpDirectory(), test_dir_, "multitasker");
    model::ModelIOOptions save_model_io;
    save_model_io.file_prefix = "my_prefix";
    EXPECT_OK(model::SaveModel(prefix_model_path, model_.get(), save_model_io));

    // Loading the multitasker model.
    std::unique_ptr<model::AbstractModel> loaded_model;
    model::ModelIOOptions load_model_io;
    load_model_io.file_prefix = "my_prefix";
    EXPECT_OK(
        model::LoadModel(prefix_model_path, &loaded_model, load_model_io));

    // Loading a submodel directly.
    std::unique_ptr<model::AbstractModel> loaded_submodel;
    model::ModelIOOptions load_submodel_io;
    load_submodel_io.file_prefix = "my_prefix_1";
    EXPECT_OK(model::LoadModel(prefix_model_path, &loaded_submodel,
                               load_submodel_io));
    EXPECT_EQ(loaded_submodel->label(), "age");
  }
}

}  // namespace
}  // namespace multitasker
}  // namespace model
}  // namespace yggdrasil_decision_forests
