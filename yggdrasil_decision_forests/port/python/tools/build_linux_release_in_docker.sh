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


DOCKER=gcr.io/tfx-oss-public/manylinux2014-bazel:bazel-5.3.0

# Current directory
# Useful if Yggdrasil Decision Forests is available locally in a neighbor
# directory.
YDF_PATH=$(realpath $PWD/../../..)
YDF_DIRNAME=${YDF_PATH##*/}

# Download docker
sudo docker pull ${DOCKER}

# Start docker
sudo docker run -it -v ${PWD}/../../../../:/working_dir -w /working_dir/${YDF_DIRNAME}/yggdrasil_decision_forests/port/python ${DOCKER} \
  /bin/bash -c "./tools/build_linux_release.sh"
