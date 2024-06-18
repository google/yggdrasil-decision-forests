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

// Evaluation of a model on a dataset stored in disk.

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_EVALUATE_ON_DISK_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_EVALUATE_ON_DISK_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests::model {

// Evaluates the model on a dataset stored in disk. `typed_path` defines
// the type and the path pattern of the files, as described in
// `yggdrasil_decision_forests/datasets/format.h` file.
// This method is preferable when the number of examples is large since they
// do not have to be all first loaded into memory.
// Returns a finalized EvaluationResults.
// Evaluates the model on a dataset. Returns a finalized EvaluationResults.
// The random generator "rnd" is used bootstrapping of confidence intervals
// and sub-sampling evaluation (if configured in "option").
absl::StatusOr<metric::proto::EvaluationResults> EvaluateOnDisk(
    const AbstractModel& model, const absl::string_view typed_path,
    const metric::proto::EvaluationOptions& option, utils::RandomEngine* rnd);

}  // namespace yggdrasil_decision_forests::model

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_EVALUATE_ON_DISK_H_
