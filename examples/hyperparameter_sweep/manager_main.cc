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

#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"

#include "examples/hyperparameter_sweep/optimizer.pb.h"

namespace ydf = ::yggdrasil_decision_forests;

// Flags of the binary.

ABSL_FLAG(std::string, work_dir, "",
          "Working directory. A directory path shared by the manager and the "
          "workers. NOTE: Currently, this directory is not used.");
ABSL_FLAG(std::string, output_dir, "",
          "Directory containing the results (i.e. the json result file).");

ABSL_FLAG(std::string, dataset_train, "",
          "Path to the training dataset with a type prefix e.g. "
          "\"csv:/project/dataset.csv\"");

ABSL_FLAG(std::string, dataset_test, "",
          "Path to the testing dataset with a type prefix.");

ABSL_FLAG(std::string, distribute_config, {},
          "Text serialized ydf::distribute::proto::Config configuring the "
          "computing resources (e.g. multi-threading, remote workers).");

ABSL_FLAG(std::string, label, "",
          "Name of the label column in the training and testing dataset.");

ABSL_FLAG(
    int, max_num_runs, -1,
    "If >0, max_num_runs is the maximum number of trained model. Use a small "
    "value of max_num_runs (e.g. max_num_runs=5) during development.");

ABSL_FLAG(int, num_repetitions, 5,
          "Number of time each hyper-parameter configuration is tested.");

ABSL_FLAG(
    int, parallel_execution_per_worker, 1,
    "Number of model trained/evaluated/benchmarked in parallel on each worker. "
    "Training multiple models in parallel on a worker is more efficient. "
    "However, it can also bias the inference speed estimation results.");

namespace example {

absl::Status Run() {
  // Create the work directory
  RETURN_IF_ERROR(file::RecursivelyCreateDir(absl::GetFlag(FLAGS_work_dir),
                                             file::Defaults()));
  RETURN_IF_ERROR(file::RecursivelyCreateDir(absl::GetFlag(FLAGS_output_dir),
                                             file::Defaults()));

  // Results in csv files.
  ASSIGN_OR_RETURN(auto file_handle,
                   file::OpenOutputFile(file::JoinPath(
                       absl::GetFlag(FLAGS_output_dir), "results.json")));
  file::OutputFileCloser results(std::move(file_handle));
  RETURN_IF_ERROR(results.stream()->Write("[\n"));

  // Path to the training and testing dataset.
  const auto train_dataset_path = absl::GetFlag(FLAGS_dataset_train);
  const auto test_dataset_path = absl::GetFlag(FLAGS_dataset_test);

  YDF_LOG(INFO) << "Initialize worker manager";
  proto::Initialization initialize;
  initialize.set_train_path(train_dataset_path);
  initialize.set_test_path(test_dataset_path);

  ASSIGN_OR_RETURN(auto distribute_config,
                   ydf::utils::ParseTextProto<ydf::distribute::proto::Config>(
                       absl::GetFlag(FLAGS_distribute_config)));
  if (!distribute_config.has_implementation_key()) {
    distribute_config.set_implementation_key("MULTI_THREAD");
  }

  ASSIGN_OR_RETURN(auto manager,
                   ydf::distribute::CreateManager(
                       distribute_config,
                       /*worker_name=*/"HYPER_PARAMETER_SWEEPER",
                       /*welcome_blob=*/initialize.SerializeAsString(),
                       /*parallel_execution_per_worker=*/
                       absl::GetFlag(FLAGS_parallel_execution_per_worker)));

  int num_commands = 0;
  const int max_num_runs = absl::GetFlag(FLAGS_max_num_runs);

  // Schedule model trainings.
  //
  // Those for-loops define the hyper-parameter sweep.
  for (int num_trees : {5, 10, 20, 30, 40, 50, 60})
    for (float shrinkage : {0.1, 0.2, 0.3})
      for (float subsample : {0.1, 0.2, 0.3})
        for (bool use_hessian_gain : {true, false})
          for (bool global_growth : {true, false}) {
            // ======
            for (int max_depth : {-1, 4, 6, 8, 10})
              for (int max_nodes : {-1, 32, 64, 80, 128, 200, 256}) {
                // ============

                if (global_growth) {
                  if (max_depth != -1) continue;
                  if (max_nodes == -1) continue;
                } else {
                  if (max_depth == -1) continue;
                  if (max_nodes != -1) continue;
                }

                if (max_num_runs >= 0 && num_commands >= max_num_runs) {
                  break;
                }

                proto::Request request;
                request.set_num_repetitions(
                    absl::GetFlag(FLAGS_num_repetitions));
                request.set_run_idx(num_commands);
                *request.mutable_param_json() =
                    absl::Substitute(R"("algorithm": "GRADIENT_BOOSTED_TREES",
"num_trees": $0,
"shrinkage": $1,
"subsample": $2,
"use_hessian_gain": $3,
"global_growth": $4,
"max_depth": $5,
"max_nodes": $6,
"run_idx": $7,
)",
                                     num_trees,         // 0
                                     shrinkage,         // 1
                                     subsample,         // 2
                                     use_hessian_gain,  // 3
                                     global_growth,     // 4
                                     max_depth,         // 5
                                     max_nodes,         // 6
                                     num_commands       // 7
                    );
                auto& train_config = *request.mutable_train_config();
                train_config.set_learner("GRADIENT_BOOSTED_TREES");
                train_config.set_task(ydf::model::proto::Task::CLASSIFICATION);
                train_config.set_label(absl::GetFlag(FLAGS_label));
                auto& gbt_config = *train_config.MutableExtension(
                    ydf::model::gradient_boosted_trees::proto::
                        gradient_boosted_trees_config);
                gbt_config.set_validation_set_ratio(0);

                gbt_config.set_num_trees(num_trees);
                gbt_config.set_shrinkage(shrinkage);
                gbt_config.set_subsample(subsample);
                gbt_config.set_use_hessian_gain(use_hessian_gain);
                if (global_growth) {
                  gbt_config.mutable_decision_tree()
                      ->mutable_growing_strategy_best_first_global()
                      ->set_max_num_nodes(max_nodes);
                } else {
                  gbt_config.mutable_decision_tree()->set_max_depth(max_depth);
                }

                RETURN_IF_ERROR(manager->AsynchronousProtoRequest(request));
                num_commands++;

                // ========
              }
          }

  YDF_LOG(INFO) << "Commands: " << num_commands;

  int solved_commands = 0;

  // Collect results.
  bool first_result = true;
  while (solved_commands < num_commands) {
    ASSIGN_OR_RETURN(const auto result,
                     manager->NextAsynchronousProtoAnswer<proto::Result>());
    solved_commands++;
    YDF_LOG(INFO) << "Receive result " << solved_commands << " / "
                  << num_commands;

    for (const auto& result_item : result.items()) {
      // Comma in json file.
      if (first_result) {
        first_result = false;
      } else {
        RETURN_IF_ERROR(results.stream()->Write(",\n"));
      }

      RETURN_IF_ERROR(results.stream()->Write(absl::Substitute(
          R"({
$0
"accuracy": $1,
"aucs": [$2],
"time_per_predictions_s": $3,
"repetition_idx": $4
}
)",
          result_item.param_json(),                // 0
          result_item.accuracy(),                  // 1
          absl::StrJoin(result_item.aucs(), ","),  // 2
          result_item.time_per_predictions_s(),    // 3
          result_item.repetition_idx()             // 4
          )));
    }
  }

  RETURN_IF_ERROR(results.stream()->Write("]\n"));
  RETURN_IF_ERROR(manager->Done());
  return absl::OkStatus();
}

}  // namespace example

int main(int argc, char** argv) {
  InitLogging(argv[0], &argc, &argv, true);

  CHECK_OK(example::Run());

  return 0;
}
