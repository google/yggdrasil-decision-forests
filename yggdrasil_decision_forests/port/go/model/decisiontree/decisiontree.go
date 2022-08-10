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

// Package decisiontree contains utilities to handle decision trees.
package decisiontree

import (
	"fmt"

	pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/decisiontree/proto"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/decisiontree/io"

	// Include I/O support for standard formats.
	_ "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/decisiontree/io/canonical"
)

// DefaultNodeFilename is the default filename to store nodes.
const DefaultNodeFilename = "nodes"

// Node is a tree node.
type Node struct {
	RawNode       *pb.Node
	PositiveChild *Node
	NegativeChild *Node
}

// IsLeaf tests if a node is a leaf.
func (n *Node) IsLeaf() bool {
	return n.PositiveChild == nil
}

// Tree is a decision tree.
type Tree struct {
	// Root node of the tree. nil if the tree is empty.
	Root *Node
}

// Forest is a collection of trees.
type Forest struct {
	Trees []*Tree
}

// NumLeafs is the number of leafs in a sub-tree.
func (n *Node) NumLeafs() int {
	if n.IsLeaf() {
		return 1
	}
	return n.PositiveChild.NumLeafs() + n.NegativeChild.NumLeafs()
}

// NumNonLeafs is the number of non-leaf nodes in a sub-tree.
func (n *Node) NumNonLeafs() int {
	if n.IsLeaf() {
		return 0
	}
	return 1 + n.PositiveChild.NumNonLeafs() + n.NegativeChild.NumNonLeafs()
}

// NumLeafs is the number of leafs in the forest.
func (f *Forest) NumLeafs() int {
	count := 0
	for _, tree := range f.Trees {
		if tree.Root != nil {
			count += tree.Root.NumLeafs()
		}
	}
	return count
}

// NumNonLeafs is the number of non-leaf nodes in the forest.
func (f *Forest) NumNonLeafs() int {
	count := 0
	for _, tree := range f.Trees {
		if tree.Root != nil {
			count += tree.Root.NumNonLeafs()
		}
	}
	return count
}

func newNode(reader io.Reader) (*Node, error) {
	// Read a new node
	rawNode, err := reader.Next()
	if err != nil {
		return nil, err
	}
	if rawNode == nil {
		// No more nodes
		return nil, fmt.Errorf("Not enough nodes")
	}

	node := &Node{RawNode: rawNode}

	if rawNode.GetCondition() != nil {
		// Read the two child nodes
		node.NegativeChild, err = newNode(reader)
		if err != nil {
			return nil, err
		}

		node.PositiveChild, err = newNode(reader)
		if err != nil {
			return nil, err
		}
	}

	return node, nil
}

// LoadForest loads a forest from disk.
func LoadForest(basePath string, numShards int, format string, numTrees int) (*Forest, error) {

	forest := &Forest{}
	forest.Trees = make([]*Tree, 0)

	reader, err := io.NewNodeReader(basePath, numShards, format)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	for treeIdx := 0; treeIdx < numTrees; treeIdx++ {
		root, err := newNode(reader)
		if err != nil {
			return nil, fmt.Errorf("decisiontree.LoadForest() after reading %d nodes, got error: %w", treeIdx, err)
		}
		forest.Trees = append(forest.Trees, &Tree{Root: root})
	}

	return forest, nil
}
