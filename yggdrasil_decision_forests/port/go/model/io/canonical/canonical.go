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

// Package canonical is an alias for the `model/io` package that also links
// all the canonical model support along. Most of the time one wants to use
// this package instead of `model/io`. But to decrease code bloat, one can
// also depend on `model/io` and only the specific type of model desired.
package canonical

import (
	model_io "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/io"

	// Include "canonical" model support.
	_ "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/canonical"
)

var (
	// LoadModel loads a model from disk.
	// This is just an alias, see implementation in `model/io.go`
	LoadModel = model_io.LoadModel

	// LoadModelWithPrefix loads a model with a prefix from disk.
	// This is just an alias, see implementation in `model/io.go`
	//
	// The "prefix" is a string append to the name of all the files in the model. Using a prefix make
	// it possible to store multiple models in the same directory without sub-directories. Models
	// created by TensorFlow Decision Forests generally have prefix.
	LoadModelWithPrefix = model_io.LoadModelWithPrefix

	// DetectFilePrefix detect the prefix of the model.
	// This function is similar as "DetectFilePrefix" in `model_library.cc`.
	// This is just an alias, see implementation in `model/io.go`
	DetectFilePrefix = model_io.DetectFilePrefix
)
