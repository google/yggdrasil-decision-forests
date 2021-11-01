#!/bin/bash
# Copyright 2021 Google LLC.
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


# Pack the CLI binary in an archive.

set -e
set -x

CLI="bazel-bin/yggdrasil_decision_forests/cli"

cp -f configure/cli_readme.txt ${CLI}/README
cp -f documentation/cli.txt ${CLI}/

pushd ${CLI}
zip -j cli_linux.zip \
  README \
  cli.txt \
  train \
  show_model \
  show_dataspec \
  predict \
  infer_dataspec \
  evaluate \
  convert_dataset \
  benchmark_inference \
  utils/synthetic_dataset \
  ../utils/distribute/implementations/grpc/grpc_worker_main
popd

mkdir -p dist
mv ${CLI}/cli_linux.zip dist/
zip ${CLI}/cli_linux.zip LICENSE CHANGELOG.md
