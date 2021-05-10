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

// Create a synthetic dataset.
//
// See yggdrasil_decision_forests/dataset/synthetic_dataset.cc for a description
// of the algorithm used to generate the dataset.
//
// Basic usage example:
//
//   bazel run -c opt :synthetic_dataset -- \
//     --alsologtostderr \
//     --train=csv:/path/to/my/train.csv \
//     --test=csv:/path/to/my/test.csv \
//     --ratio_test=0.2
//
// Advanced usage example:
//
//   // Content of "config.pbtxt"
//   num_examples: 10000
//   missing_ratio: 0.20
//   categorical_vocab_size: 100
//   num_multidimensional_numerical: 10
//
//   bazel run -c opt :synthetic_dataset -- \
//     --alsologtostderr \
//     --options=config.pbtxt\
//     --train=csv:/path/to/my/train.csv \
//     --valid=csv:/path/to/my/valid.csv \
//     --test=csv:/path/to/my/test.csv \
//     --ratio_valid=0.2 \
//     --ratio_test=0.2
//
#include "yggdrasil_decision_forests/dataset/synthetic_dataset.h"

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, options, "",
          "Optional path to text serialized proto::SyntheticDatasetOptions.");

ABSL_FLAG(std::string, train, "",
          "[type]:[path] path to the output training dataset.");

ABSL_FLAG(std::string, valid, "",
          "Optional [type]:[path] path to the output validation dataset.");

ABSL_FLAG(std::string, test, "",
          "Optional [type]:[path] path to the output test dataset.");

ABSL_FLAG(float, ratio_valid, 0.f,
          "Fraction of the dataset (which size is defined in \"options\") is "
          "send to the validation dataset. The \"valid\" flag can be empty "
          "iff. ratio_valid=0.");

ABSL_FLAG(
    float, ratio_test, 0.3f,
    "Fraction of the dataset (which size is defined in \"options\") is send to "
    "the test dataset. The \"test\" flag can be empty iff. ratio_valid=0.");

constexpr char kUsageMessage[] = "Create a synthetic dataset.";

namespace yggdrasil_decision_forests {
namespace dataset {

void SyntheticDataset() {
  proto::SyntheticDatasetOptions options;
  if (!absl::GetFlag(FLAGS_options).empty()) {
    QCHECK_OK(file::GetTextProto(absl::GetFlag(FLAGS_options), &options,
                                 file::Defaults()));
  }
  LOG(INFO) << "Options:\n" << options.DebugString();
  QCHECK_OK(GenerateSyntheticDatasetTrainValidTest(
      options, absl::GetFlag(FLAGS_train), absl::GetFlag(FLAGS_valid),
      absl::GetFlag(FLAGS_test), absl::GetFlag(FLAGS_ratio_valid),
      absl::GetFlag(FLAGS_ratio_test)));
  LOG(INFO) << "Generation done";
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::dataset::SyntheticDataset();
  return 0;
}
