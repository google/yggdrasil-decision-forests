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

// Package io contains utilities to save and load models. It doesn't include any actual model
// type support by default. Consider using instead the subpackage `canonical` that includes
// the canonical (standard) model types support.
package io

import (
	"context"
	"fmt"
	"path/filepath"

	dataspec_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/dataset/proto"
	model_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/proto"

	// External dependencies, pls keep in this position in file.
	"google.golang.org/protobuf/proto"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/utils/file"
	// End of external dependencies.

	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model"
)

// Specific model filenames.
const modelHeaderFileName = "header.pb"
const modelDataSpecFileName = "data_spec.pb"

// LoadModel loads a model from disk.
func LoadModel(modelPath string) (model.Model, error) {
	prefix, err := DetectFilePrefix(modelPath)
	if err != nil {
		return nil, err
	}
	return LoadModelWithPrefix(modelPath, prefix)
}

// LoadModelWithPrefix loads a model with a prefix from disk.
//
// The "prefix" is a string append to the name of all the files in the model. Using a prefix make
// it possible to store multiple models in the same directory without sub-directories. Models
// created by TensorFlow Decision Forests generally have prefix.
func LoadModelWithPrefix(modelPath string, prefix string) (model.Model, error) {
	// Read the generic header.
	serializedHeader, err := file.ReadFile(context.Background(), filepath.Join(modelPath, prefix+modelHeaderFileName))
	if err != nil {
		return nil, err
	}
	header := &model_pb.AbstractModel{}
	if err := proto.Unmarshal(serializedHeader, header); err != nil {
		return nil, err
	}

	// Read the dataspec.
	serializedDataspec, err := file.ReadFile(context.Background(), filepath.Join(modelPath, prefix+modelDataSpecFileName))
	if err != nil {
		return nil, err
	}
	dataspec := &dataspec_pb.DataSpecification{}
	if err := proto.Unmarshal(serializedDataspec, dataspec); err != nil {
		return nil, err
	}

	// Instantiate the model object.
	builder, hasBuilder := model.RegisteredBuilders[header.GetName()]
	if !hasBuilder {
		return nil, fmt.Errorf(
			"unknown model %q. The available models are: %v. This may be because this type of model "+
				"was not imported -- directly or through the \"canonical\" package that automatically "+
				"imports all implemented models -- or it is not implemented in Go yet, in which case feel "+
				"free to reach out the maintainers in github and ask for it",
			header.GetName(), model.RegisteredBuilders)
	}

	// Load the model specific content.
	model := builder(header, dataspec)
	if err = model.LoadSpecific(modelPath, prefix); err != nil {
		return nil, err
	}

	return model, nil
}

// DetectFilePrefix detect the prefix of the model.
// This function is similar as "DetectFilePrefix" in `model_library.cc`.
func DetectFilePrefix(modelPath string) (string, error) {
	files, err := file.Match(context.Background(), filepath.Join(modelPath, "*"+modelDataSpecFileName), file.StatNone)
	if err != nil {
		return "", err
	}
	if len(files) != 1 {
		return "", fmt.Errorf("file prefix cannot be autodetected: %v models exist in %v. A model directory should contain a filename finishing by \"data_spec.pb\". Note: If you trained this model with TensorFlow Decision Forests, the model is in the \"assets\" sub-directory",
			len(files), modelPath)
	}
	dataspecFilename := filepath.Base(files[0].Path)
	return dataspecFilename[:len(dataspecFilename)-len(modelDataSpecFileName)], nil
}
