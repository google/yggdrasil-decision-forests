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


#
# Builds a binary release (like "build_binary_release.sh") in a Docker
# container.
#
# Usage example:
#   ./tools/build_binary_release.sh

set -vex

DOCKER=ubuntu:20.04

# Current directory
DIRNAME=${PWD##*/}

# Download docker
docker pull ${DOCKER}

# Start docker
docker run -it -v ${PWD}/..:/working_dir -w /working_dir/${DIRNAME} ${DOCKER} \
  /bin/bash -c "INSTALL_DEPENDENCIES=1 BUILD=1 PACK=1 ./tools/build_binary_release.sh"
