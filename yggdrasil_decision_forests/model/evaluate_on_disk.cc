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

#include "yggdrasil_decision_forests/model/evaluate_on_disk.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/dataset/weight.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests::model {

namespace {

// Evaluates a model and add the evaluation to an already initialized evaluation
// proto.
absl::Status AppendEvaluation(const AbstractModel& model,
                              const absl::string_view typed_path,
                              const metric::proto::EvaluationOptions& option,
                              utils::RandomEngine* rnd,
                              metric::proto::EvaluationResults* eval) {
  dataset::proto::LinkedWeightDefinition weight_links;
  if (option.has_weights()) {
    RETURN_IF_ERROR(dataset::GetLinkedWeightDefinition(
        option.weights(), model.data_spec(), &weight_links));
  }

  auto engine_or_status = model.BuildFastEngine();
  if (engine_or_status.ok()) {
    const auto engine = std::move(engine_or_status.value());
    // Extract the shards from the dataset path.
    std::string path, prefix;
    std::tie(prefix, path) = dataset::SplitTypeAndPath(typed_path).value();
    std::vector<std::string> shards;
    RETURN_IF_ERROR(utils::ExpandInputShards(path, &shards));

    // Evaluate each shard in a separate thread.
    utils::concurrency::Mutex
        mutex;  // Guards "num_evaluated_shards" and "eval".
    int num_evaluated_shards = 0;
    absl::Status worker_status;

    const auto process_shard = [&option, eval, &mutex, &prefix, &engine,
                                &weight_links, &num_evaluated_shards, &shards,
                                &model](absl::string_view shard,
                                        int sub_rnd_seed) -> absl::Status {
      utils::RandomEngine sub_rnd(sub_rnd_seed);

      dataset::VerticalDataset dataset;
      RETURN_IF_ERROR(dataset::LoadVerticalDataset(
          absl::StrCat(prefix, ":", shard), model.data_spec(), &dataset));

      metric::proto::EvaluationResults sub_evaluation;
      RETURN_IF_ERROR(metric::InitializeEvaluation(
          option, model.LabelColumnSpec(), &sub_evaluation));

      RETURN_IF_ERROR(model.AppendEvaluationWithEngine(
          dataset, option, weight_links, *engine, &sub_rnd, nullptr,
          &sub_evaluation));

      utils::concurrency::MutexLock lock(&mutex);
      RETURN_IF_ERROR(metric::MergeEvaluation(option, sub_evaluation, eval));
      num_evaluated_shards++;
      LOG_EVERY_N_SEC(INFO, 30) << num_evaluated_shards << "/" << shards.size()
                                << " shards evaluated";
      return absl::OkStatus();
    };

    {
      const int num_threads = std::min<int>(shards.size(), 20);
      utils::concurrency::ThreadPool thread_pool(
          num_threads, {.name_prefix = std::string("evaluation")});
      thread_pool.StartWorkers();
      for (const auto& shard : shards) {
        thread_pool.Schedule([&shard, &mutex, &process_shard, &worker_status,
                              sub_rnd_seed = (*rnd)()]() -> void {
          {
            utils::concurrency::MutexLock lock(&mutex);
            if (!worker_status.ok()) {
              return;
            }
          }
          auto sub_status = process_shard(shard, sub_rnd_seed);
          {
            utils::concurrency::MutexLock lock(&mutex);
            worker_status.Update(sub_status);
          }
        });
      }
    }

    RETURN_IF_ERROR(worker_status);

  } else {
    // Evaluate using the (slow) generic inference.
    LOG(WARNING)
        << "Evaluation with the slow generic engine without distribution";
    dataset::VerticalDataset dataset;
    RETURN_IF_ERROR(
        dataset::LoadVerticalDataset(typed_path, model.data_spec(), &dataset));
    RETURN_IF_ERROR(model.AppendEvaluation(dataset, option, rnd, eval));
    return absl::OkStatus();
  }

  eval->set_num_folds(eval->num_folds() + 1);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<metric::proto::EvaluationResults> EvaluateOnDisk(
    const AbstractModel& model, const absl::string_view typed_path,
    const metric::proto::EvaluationOptions& option, utils::RandomEngine* rnd) {
  if (option.task() != model.task()) {
    STATUS_FATAL("The evaluation and the model tasks differ.");
  }
  metric::proto::EvaluationResults eval;
  RETURN_IF_ERROR(
      metric::InitializeEvaluation(option, model.LabelColumnSpec(), &eval));
  RETURN_IF_ERROR(AppendEvaluation(model, typed_path, option, rnd, &eval));
  RETURN_IF_ERROR(
      metric::FinalizeEvaluation(option, model.LabelColumnSpec(), &eval));
  return eval;
}

}  // namespace yggdrasil_decision_forests::model
