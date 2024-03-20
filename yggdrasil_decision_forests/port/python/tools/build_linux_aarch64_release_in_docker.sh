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


DOCKER=quay.io/pypa/manylinux2014_aarch64@sha256:50312f29ac68e4234911f634228f48aafbb0846959596138c2206bbb48c576d0

BAZELISK_VERSION="v1.19.0"
YDF_PATH=$(realpath $PWD/../../..)

# Download docker
docker pull $DOCKER

# Start the container
docker run -it -v $YDF_PATH:/working_dir -w /working_dir/yggdrasil_decision_forests/port/python \
  $DOCKER /bin/bash -c " \
  yum update && yum install -y rsync && \
  curl -L -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-arm64 && \
  chmod +x /usr/local/bin/bazel && \
  ./tools/build_linux_release.sh "
