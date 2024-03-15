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


DOCKER=quay.io/pypa/manylinux2014_x86_64@sha256:2e37241d9c9fbbccea009e59505a1384f9501a7bfea77b21fdcbf332c7036e70

BAZELISK_VERSION="v1.19.0"
YDF_PATH=$(realpath $PWD/../../..)

# Download docker
docker pull $DOCKER

# Start the container
docker run -it -v $YDF_PATH:/working_dir -w /working_dir/yggdrasil_decision_forests/port/python \
  $DOCKER /bin/bash -c " \
  yum update && yum install -y rsync && \
  curl -L -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64 && \
  chmod +x /usr/local/bin/bazel && \
  ./tools/build_linux_release.sh "
