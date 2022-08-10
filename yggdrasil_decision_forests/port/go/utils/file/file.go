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

// Package file is an incomplete slim portability layer between Google internal file libraries and "os" package.
package file

import (
	"context"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
)

// File for providing the shim layer
type File struct {
	file *os.File
}

// Stat holds information about a file.
type Stat struct {
	// Path to the file.
	Path string
	// Missing other fields.
}

// StatMask specifies what fields should be returned. Not implemented.
type StatMask int

// Known values for StatMask.
const (
	StatNone StatMask = 0
)

// Create a file
func Create(ctx context.Context, name string) (File, error) {
	file, err := os.Create(name)
	return File{file: file}, err
}

// IO is a convenience interface.
type IO io.ReadCloser

// IO to get the os.File member
func (f *File) IO(ctx context.Context) IO {
	return f.file
}

// ReadFile returns the entire contents of the named file.
func ReadFile(ctx context.Context, name string) ([]byte, error) {
	return ioutil.ReadFile(name)
}

// WriteFile writes data to a file named by filename.
func WriteFile(ctx context.Context, name string, data []byte) error {
	return ioutil.WriteFile(name, data, 0644)
}

// MkdirOptions for creating directories. It should be a protobuf, but since
// we aren't passing it ever, we can keep it as it is, for the moment.
type MkdirOptions struct{}

// Mkdir is documented in the package-level Mkdir function.
func Mkdir(ctx context.Context, name string, perm *MkdirOptions) error {
	return os.Mkdir(name, 0755)
}

// MkdirAll is documented in the package-level MkdirAll function.
func MkdirAll(ctx context.Context, name string, perm *MkdirOptions) error {
	return os.MkdirAll(name, 0755)
}

// Chmod changes the mode of the named file.
func Chmod(ctx context.Context, name string, mode os.FileMode) error {
	return os.Chmod(name, mode)
}

// OpenRead opens the file for reading.
func OpenRead(ctx context.Context, name string) (File, error) {
	file, err := os.Open(name)
	return File{file: file}, err
}

// Match returns information about all files matching pattern. mask is ignored for now.
func Match(ctx context.Context, pattern string, mask StatMask) ([]Stat, error) {
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	stats := make([]Stat, 0, len(files))
	for _, f := range files {
		stats = append(stats, Stat{Path: f})
	}
	return stats, nil
}
