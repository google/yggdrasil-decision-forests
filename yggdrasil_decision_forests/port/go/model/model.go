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

// Package model defines the "Model" interface.
package model

import (
	dataspec_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/dataset/proto"
	model_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/proto"
)

// Model is a generic model interface.
//
// Examples:
//
// // Create a new GBT model
// model := gradientboostedtrees.Create(...)
//
// // Load an existing model.
// model, err := io.Load("/path/to/model")
// fmt.Println("My model is a %v.", model.Name())
// >> My model is a GRADIENT_BOOSTED_TREES.
type Model interface {

	// Registered name of the model.
	Name() string

	// Header of the model.
	Header() *model_pb.AbstractModel

	// Dataspec of the model.
	Dataspec() *dataspec_pb.DataSpecification
}

// Implementation interface needs to be implemented by Models (mostly internal).
// Not needed by those only using a model.
type Implementation interface {
	Model

	// LoadSpecific loads the model implementation specific data from a directory.
	// Users are not expected to call this method directly. Instead, models
	// should be loaded with "model, err := io.LoadModel(path)".
	LoadSpecific(modelPath string, prefix string) error
}

// RegisteredBuilders is the list of model builders, keyed by a unique `ModelKey` string per model type.
// Only register (change) this during the runtime initialization, in `init()` function.
// End users probably want to use `io.LoadModel()` to load models instead.
var RegisteredBuilders = make(map[string]func(header *model_pb.AbstractModel, dataspec *dataspec_pb.DataSpecification) Implementation)
