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

// Package gradientboostedtrees defines the gradient boosted trees model.
package gradientboostedtrees

import (
	"context"
	"fmt"
	"path/filepath"

	dataspec_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/dataset/proto"
	model_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/proto"
	gbt_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/gradientboostedtrees/proto"
	dt "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/decisiontree"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model"

	// External dependencies, pls keep in this position in file.
	"google.golang.org/protobuf/proto"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/utils/file"
	// End of external dependencies.//
	//
)

// ModelKey is the unique identifier of the model for serizalization.
const ModelKey = "GRADIENT_BOOSTED_TREES"

// Filename containing the GBT header.
const headerFilename = "gradient_boosted_trees_header.pb"

// Model is a Gradient Boosted Trees model.
type Model struct {
	header    *model_pb.AbstractModel
	dataspec  *dataspec_pb.DataSpecification
	GbtHeader *gbt_pb.Header
	Forest    *dt.Forest
}

func init() {
	// Register the constructor (loader) for GBTs.
	model.RegisteredBuilders[ModelKey] = Create
}

// Create creates a GBT model.
func Create(header *model_pb.AbstractModel, dataspec *dataspec_pb.DataSpecification) model.Implementation {
	return &Model{header: header, dataspec: dataspec, GbtHeader: nil}
}

// Name of the model.
func (me *Model) Name() string {
	return ModelKey
}

// Header of the model.
func (me *Model) Header() *model_pb.AbstractModel {
	return me.header
}

// Dataspec of the model.
func (me *Model) Dataspec() *dataspec_pb.DataSpecification {
	return me.dataspec
}

// LoadSpecific loads a model from disk.
func (me *Model) LoadSpecific(modelPath string, prefix string) error {

	// Load the GBT specialized header.
	serializedGbtHeader, err := file.ReadFile(context.Background(),
		filepath.Join(modelPath, prefix+headerFilename))
	if err != nil {
		return err
	}
	me.GbtHeader = &gbt_pb.Header{}
	if err := proto.Unmarshal(serializedGbtHeader, me.GbtHeader); err != nil {
		return err
	}

	// Load the forest structure.
	me.Forest, err = dt.LoadForest(
		filepath.Join(modelPath, prefix+dt.DefaultNodeFilename),
		int(me.GbtHeader.GetNumNodeShards()),
		me.GbtHeader.GetNodeFormat(),
		int(me.GbtHeader.GetNumTrees()))
	if err != nil {
		return err
	}

	if len(me.Forest.Trees) != int(me.GbtHeader.GetNumTrees()) {
		return fmt.Errorf("Don't number of trees in the model")
	}
	return nil
}
