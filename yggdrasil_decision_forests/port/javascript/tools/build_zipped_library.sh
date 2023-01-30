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



# If your env supports it, adds the following:
# --config=lto
# --config=size

set -vex
bazel build -c opt --config=wasm \
  //yggdrasil_decision_forests/port/javascript:create_release

mkdir -p dist
cp -f bazel-bin/yggdrasil_decision_forests/port/javascript/ydf.zip dist/ydf.zip
cp -f bazel-bin/yggdrasil_decision_forests/port/javascript/ydf.zip dist/javascript_wasm.zip
