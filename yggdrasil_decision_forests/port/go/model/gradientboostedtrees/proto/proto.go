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

// yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.proto proto compilation to Go:
//go:generate protoc -I. -I../../../../../.. --go_out=../../../../../../yggdrasil_decision_forests/port/go --go_opt=paths=import --go_opt=module=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go  --go_opt=Myggdrasil_decision_forests/dataset/data_spec.proto=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/dataset/proto  --go_opt=Myggdrasil_decision_forests/model/abstract_model.proto=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/proto  --go_opt=Myggdrasil_decision_forests/model/decision_tree/decision_tree.proto=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/decisiontree/proto  --go_opt=Myggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.proto=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/gradientboostedtrees/proto  --go_opt=Myggdrasil_decision_forests/model/random_forest/random_forest.proto=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/randomforest/proto  --go_opt=Myggdrasil_decision_forests/utils/distribution.proto=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/utils/proto  --go_opt=Myggdrasil_decision_forests/dataset/weight.proto=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/dataset/proto  --go_opt=Myggdrasil_decision_forests/model/hyperparameter.proto=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/proto  --go_opt=Myggdrasil_decision_forests/metric/metric.proto=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/metric/proto  --go_opt=Myggdrasil_decision_forests/model/prediction.proto=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/proto --go_opt=Mgradient_boosted_trees.proto=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/gradientboostedtrees/proto yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.proto

// File automatically generated, please don't edit it directly.

// Use `sudo apt install protobuf-compiler protoc-gen-go` to install the protobuf compiler.
// Alternatively use `go install google.golang.org/protobuf/cmd/protoc-gen-go@latest` to get latest suproto_dirort for Go.

// Package proto includes all proto definitions used in the golang package in one large package.
//
// It uses go generate tools to generate it from the source code, but we include the generated
// files in github, so one doesn't need to install anything.
package proto
