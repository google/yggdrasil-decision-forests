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

package main

import (
	"testing"
)

// TestBenchmark evaluates inference speed on a well known model/dataset. At the time of
// the writing of this test (Q3 2022), in a "normal" desktop the avg. time per example was
// around 6.4 microseconds.
func TestBenchmark(t *testing.T) {
	options := Options{
		numRuns:    5,
		batchSize:  100,
		warmupRuns: 2}
	err := Run(
		"../../../../test_data/model/adult_binary_class_gbdt/",
		"csv:../../../../test_data/dataset/adult.csv",
		&options)

	if err != nil {
		t.Fatal(err)
	}
}
