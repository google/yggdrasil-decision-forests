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

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/serving/embed/test_model1.h"
#include "yggdrasil_decision_forests/serving/embed/test_model2.h"

namespace yggdrasil_decision_forests::serving::embed {
namespace {

TEST(Embed, Example) {
  test_model1::f();
  test_model2::f();
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed
