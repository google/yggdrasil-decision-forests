#!/bin/bash
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Generates the wrapper proto packages for the open-source Go distribution.
# These generated packages (`.../proto/proto.go` directories+files) contain 
# only the `go:generate` tags that will later generate the Go code 
# (`.../proto/*.pb.go` files) for the protos used.
#
# Notice one can also create a `.../proto/proto.go` manually. It just takes
# some carefully selected paths. 
#
# This script is meant only internally to Google, at least for now, since
# its highly dependent on the directories exported to copybara. External
# users can create the `proto.go` files manually.
#
# This script should be run only when a new proto is used (or depended on)
# by the Go OSS port. The small wrapper it creates could be created manually,
# but it's an error prone process, hence the script.
#
# Requrements
#
#   - Go
#   - Protocol buffer for Go:  protoc
#   - Go plugins for the protocol compiler
#
# Steps to run it:
#
# 1. Go (`cd`) the citc client, under `google3/third_party/yggdrasil_decision_forests/port/go`.
#    directory.
#
#    $ cd third_party/yggdrasil_decision_forests/port/go
#
# 2. Execute the script. It will generate the corresponding directories and
#    thin wrapper go code.
# 3. Create/snapshot a CL with the changes and upload it to Critique. Save the
#    CL number.
#
#    $ CL=<the actual CL number>
#
# 4. Go back (`cd`) to `google3` and run copybara to generate a local OSS
#    version with the new code.
#
#    $ rm -rf "${HOME}/git/yggdrasil-decision-forests" \
#      && /google/bin/releases/copybara/public/copybara/copybara \
#         ../../../../third_party/yggdrasil_decision_forests/copy.bara.sky \
#         presubmit_piper_to_gerrit "${CL}" --dry-run --init-history --squash \
#         --force --git-destination-path ${HOME}/git/yggdrasil-decision-forests\
#         --ignore-noop
# 
#    And then go that directory, under the golang port -- typically this will be
#    `cd ~/git/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go`
#
# 5. In each of the proto directories generated, run `go generate` to actually
#    generate the proto handling code (`*.pb.go` files). One can do:
#
#     $ find . -type d -name proto \
#       | while read dd ; do \
#          echo $dd ; pushd ${dd} ; go generate ; popd ; printf "\n\n\n" ; \
#         done   
#
# 6. Check that the generated file compiles, to double check. Either with
#    `go build ./...` or more specifically:
# 
#     $ find . -type d -name proto \
#       | while read dd ; do \
#          echo $dd ; pushd ${dd} ; go build ; popd ; printf "\n\n\n" ; \
#         done   
#
# 7. Copy back the generated `.pb.go` files back to the citc client. This is
#    because typically the generated files are include in the repository for
#    Go packages, so that users don't need to install extra software to build.

# Check we are in the Golang port directory.
GOLANG_PORT_DIR="third_party/yggdrasil_decision_forests/port/go"
current_path="$(pwd)"
if [[ ! "${current_path}" == *"${GOLANG_PORT_DIR}" ]]; then
  echo "Please, run this script to from .../${GOLANG_PORT_DIR}. You are in ${current_path}."
  exit 1
fi

# List of proto_files used in port/go. Should includ both those imported
# explicitly and the dependencies. FutureWork: figure out dependencies 
# automatically.
proto_files="$(
	find . -type f -name '*.go' | 
		xargs egrep '_go_proto"$' | 
		perl -ne 's/.*?\s+\"(.*)\"/$1/g; s/_go_proto/.proto/g; s|google3/third_party/yggdrasil_decision_forests/||g; print;' | sort | uniq) 
	utils/distribution.proto 
	dataset/weight.proto
	model/hyperparameter.proto
  metric/metric.proto
  model/prediction.proto"

# Convert a name to Ga package name: that is, without "_".
function go_name() {
  echo $1 | tr -d '_'
}

# Extract various names of paths used in the template below.
function get_names() {
  proto_path="$1"
  proto_dir=$(dirname $proto_path)
  proto_name=$(basename $proto_path .proto)
  go_proto_name=$(go_name $proto_name)
  last_dir=$(basename $proto_dir)

  # Extract package name from proto, and create Go path from this. It assumes
  go_dir="$(cat "../../${proto_path}" \
    | egrep '^package ' \
    | perl -ne 's|^package yggdrasil_decision_forests\.(.*);$|$1|g; s|\.|/|g; print;' \
  )"
  go_dir="$(go_name "${go_dir}")"  # Directory under `ydf/port/go`
  go_proto_wrapper="${go_dir}/proto.go"
  url_go="github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/${go_dir}"
  echo $proto_dir $proto_name $go_proto_name $go_dir $url_go $go_proto_wrapper
}

# Generate the protoc for Go compilation options. These will be included in every proto,
# so we don't need to spell out the dependencies individually.
protoc_go_opt="--go_opt=paths=import --go_opt=module=github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go"
for proto_path in $proto_files ; do
  read -r proto_dir proto_name go_proto_name go_dir url_go go_proto_wrapper <<<$(get_names $proto_path)
  protoc_go_opt="${protoc_go_opt} "
  protoc_go_opt="${protoc_go_opt} --go_opt=Myggdrasil_decision_forests/${proto_path}=${url_go}"
done

echo "protoc options:"
for oo in $protoc_go_opt ; do
  echo "  ${oo}"
done
printf "\n\n"

# Template for the proto.go program, which include the go:generate tags.
read -r -d '' TEMPLATE_PER_PROTO_FILE << 'EndOfTemplate'
// yggdrasil_decision_forests/PROTO_FILE proto compilation to Go:
//go:generate protoc -I. -IROOT --go_out=ROOT/yggdrasil_decision_forests/port/go PROTOC_GO_OPT --go_opt=MPROTO_BASENAME.proto=URL_GO yggdrasil_decision_forests/PROTO_FILE
EndOfTemplate

read -r -d '' TEMPLATE_SUFFIX << 'EndOfTemplate'
// File automatically generated, please don't edit it directly.

// Use `sudo apt install protobuf-compiler protoc-gen-go` to install the protobuf compiler.
// Alternatively use `go install google.golang.org/protobuf/cmd/protoc-gen-go@latest` to get latest suproto_dirort for Go.

// Package proto includes all proto definitions used in the golang package in one large package.
//
// It uses go generate tools to generate it from the source code, but we include the generated
// files in github, so one doesn't need to install anything.
package proto
EndOfTemplate

# Create directory / reset proto.go files.
for proto_path in $proto_files ; do
  read -r proto_dir proto_name go_proto_name go_dir url_go go_proto_wrapper <<<$(get_names $proto_path)
  echo "Setting up ${go_dir}"
  mkdir -p "${go_dir}"  # Make sure target directory exists.  
  rm -f "${go_proto_wrapper}"
  touch "${go_proto_wrapper}"
done

# Execute template 1 for each proto file included.
for proto_path in $proto_files ; do
  read -r proto_dir proto_name go_proto_name go_dir url_go go_proto_wrapper <<<$(get_names $proto_path)
  echo "Appending to ${go_proto_wrapper}:"

  rel_path_to_root="../../../..$(echo "${go_dir}" | tr -d 'A-Za-z_.' | perl -ne 's|/|/..|g; print;' )"
  echo "${TEMPLATE_PER_PROTO_FILE}" \
    | perl -ne 's|ROOT|'"${rel_path_to_root}"'|g; s|PROTO_FILE|'"${proto_path}"'|g; s|PROTO_BASENAME|'"${proto_name}"'|g; s|PROTOC_GO_OPT|'"${protoc_go_opt}"'|g; s|URL_GO|'"${url_go}"'|g; print;' \
    | tee -a "${go_proto_wrapper}"
  echo >> "${go_proto_wrapper}"  # Separating line.
  printf "====================================\n\n"
done

# Write suffix.
for proto_path in $proto_files ; do
  read -r proto_dir proto_name go_proto_name go_dir url_go go_proto_wrapper <<<$(get_names $proto_path)
  if ! grep -q "package proto" "${go_proto_wrapper}" ; then
    echo "Writing suffix to ${go_proto_wrapper}"
    echo "${TEMPLATE_SUFFIX}" >> "${go_dir}/proto.go"
  fi
done
