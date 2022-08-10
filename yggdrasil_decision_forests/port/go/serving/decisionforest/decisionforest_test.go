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

package decisionforest

// Test for decisionforest.
//
// The end-to-end tests for the "decisionforest" package are in "serving/serving_test.go".

import (
	"testing"

	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/utils/test"
)

func TestGetSetBit(t *testing.T) {
	data := []byte{0x00, 0x00}

	test.CheckEq(t, GetBit(data, 4), false, "")
	SetBit(data, 4)
	test.CheckEq(t, data, []byte{1 << 4, 0x00}, "")

	test.CheckEq(t, GetBit(data, 4), true, "")

	SetBit(data, 6)
	test.CheckEq(t, data, []byte{(1 << 4) | (1 << 6), 0x00}, "")

	SetBit(data, 10)
	test.CheckEq(t, data, []byte{(1 << 4) | (1 << 6), 1 << 2}, "")
}
