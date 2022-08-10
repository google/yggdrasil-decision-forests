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

// Package blobsequence implement node reading from blob sequence files.
//
// The ground truth implementation of the blog sequence format is at
// third_party/yggdrasil_decision_forests/utils/blob_sequence.h
package blobsequence

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"

	pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/decisiontree/proto"
	nodeIO "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/decisiontree/io"

	// External dependencies, pls keep in this position in file.
	"google.golang.org/protobuf/proto"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/utils/file"
	// End of external dependencies.//
	//
)

// ModelKey is the unique identifier of the RecordIO based node format.
const ModelKey = "BLOB_SEQUENCE"

func init() {
	// Register the model.
	nodeIO.RegisteredFormats[ModelKey] = newReader
}

// blobSequenceIONodeReader is a single file reader on Blog Sequence format.
type blobSequenceIONodeReader struct {
	fileHandle *file.File
	fileIO     io.ReadCloser
	bufferedIO *bufio.Reader
	version    uint16
	// "serializedNodeBuffer" is a buffer of bytes used to parse the node proto.
	// The buffer is reused in between "Next" calls.
	serializedNodeBuffer []byte
}

func (r *blobSequenceIONodeReader) Next() (*pb.Node, error) {

	// Serialized size of the serialized node
	var length uint32
	err := binary.Read(r.bufferedIO, binary.LittleEndian, &length)
	if err == io.EOF {
		// End of sequence
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	// Resize the read buffer if necessary.
	if len(r.serializedNodeBuffer) < int(length) {
		r.serializedNodeBuffer = make([]byte, length)
	}

	// Read the serialized node
	_, err = io.ReadFull(r.bufferedIO, r.serializedNodeBuffer[:length])
	if err != nil {
		return nil, err
	}

	// Deserialize the node
	node := &pb.Node{}
	if err := proto.Unmarshal(r.serializedNodeBuffer[:length], node); err != nil {
		return nil, err
	}

	return node, nil
}

func (r *blobSequenceIONodeReader) Close() error {
	if r.fileIO != nil {
		return r.fileIO.Close()
	}
	return nil
}

// newReader creates a node reader for a blog sequence file.
func newReader(path string) (nodeIO.Reader, error) {
	ctx := context.Background()
	fileHandle, err := file.OpenRead(ctx, path)
	if err != nil {
		return nil, err
	}
	fileIO := fileHandle.IO(ctx)
	bufferedIO := bufio.NewReader(fileIO)

	// Magic number.
	// The first two bytes should be "BS" in ascii (for "blob sequence").
	var magic [2]byte
	_, err = io.ReadFull(bufferedIO, magic[:])
	if err != nil {
		return nil, err
	}
	if magic[0] != 'B' || magic[1] != 'S' {
		return nil, fmt.Errorf("Invalid header")
	}

	// Version
	var version uint16
	err = binary.Read(bufferedIO, binary.LittleEndian, &version)
	if err != nil {
		return nil, err
	}
	if version != 0 {
		// Only the version 0 is currently supported.
		return nil, fmt.Errorf("Non supported file version %d", version)
	}

	// Reserved
	var reserved uint32
	err = binary.Read(bufferedIO, binary.LittleEndian, &reserved)
	if err != nil {
		return nil, err
	}

	return &blobSequenceIONodeReader{
		fileHandle:           &fileHandle,
		fileIO:               fileIO,
		bufferedIO:           bufferedIO,
		version:              version,
		serializedNodeBuffer: make([]byte, 512),
	}, nil
}
