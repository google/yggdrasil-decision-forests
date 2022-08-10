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

// Package io contains utilities to load/save decision trees
package io

import (
	"fmt"

	pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/decisiontree/proto"
)

// Reader is a stream of nodes.
type Reader interface {

	// Next returns the next raw Node protobuffer from the stream. Returns nil at the end of the
	// stream. "Close" should be called once the reading is done (even if file reaches end).
	Next() (*pb.Node, error)
	Close() error
}

// RegisteredFormats is the list of format readers.
var RegisteredFormats = make(map[string]func(path string) (Reader, error))

// NewNodeReader creates a new node reader from a sharded set of files.
func NewNodeReader(path string, numShards int, format string) (Reader, error) {
	builder, hasBuilder := RegisteredFormats[format]
	if !hasBuilder {
		return nil, fmt.Errorf("Unknown node format %q. The available node formats are: %v)", format, RegisteredFormats)
	}
	return &shardedNodeReader{path: path, numShards: numShards,
		createSubReader: builder}, nil
}

// shardedNodeReader is a wrapper for sharded files.
type shardedNodeReader struct {
	path            string
	numShards       int
	nextShard       int
	createSubReader func(path string) (Reader, error)
	currentReader   Reader
}

func (s *shardedNodeReader) Next() (*pb.Node, error) {

	for {
		// Ensure one shard is being read
		if s.currentReader == nil {
			// The previous shard (if any) is done being read
			if s.nextShard == s.numShards {
				// No more nodes available
				return nil, nil
			}

			// Open the next shard.
			var err error
			s.currentReader, err = s.createSubReader(fmt.Sprintf("%s-%05d-of-%05d", s.path, s.nextShard, s.numShards))
			if err != nil {
				return nil, err
			}
			s.nextShard++
		}

		// Read the next example
		node, err := s.currentReader.Next()
		if err != nil {
			// Reading error
			return nil, err
		}
		if node != nil {
			// Node read
			return node, nil
		}

		// End of this shard.
		err = s.currentReader.Close()
		if err != nil {
			// Error when closing the shard
			return nil, err
		}

		s.currentReader = nil
	}
}

func (s *shardedNodeReader) Close() error {
	if s.currentReader != nil {
		return s.currentReader.Close()
	}
	return nil
}
