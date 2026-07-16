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

#include "yggdrasil_decision_forests/model/postprocessor/postprocessor_library.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/model/postprocessor/abstract_postprocessor.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests::model::postprocessor {
namespace {

using ::yggdrasil_decision_forests::test::StatusIs;

TEST(PostprocessorLibraryTest, CreatePostprocessorUnimplemented) {
  proto::AbstractPostprocessor config;
  config.set_enabled(true);

  EXPECT_THAT(
      CreatePostprocessor(config).status(),
      StatusIs(absl::StatusCode::kUnimplemented, "Not implemented yet."));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::model::postprocessor
