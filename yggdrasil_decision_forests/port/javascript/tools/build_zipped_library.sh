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
  //yggdrasil_decision_forests/port/javascript/inference:create_release

mkdir -p dist
cp -f bazel-bin/yggdrasil_decision_forests/port/javascript/inference/ydf.zip dist/ydf_inference.zip
cp -f bazel-bin/yggdrasil_decision_forests/port/javascript/inference/ydf.zip dist/javascript_wasm_inference.zip

# Extract library to NPM location
unzip dist/ydf_inference.zip -d yggdrasil_decision_forests/port/javascript/inference/npm/dist

# Remove the unnecessary readme.txt in the subfolder.
rm yggdrasil_decision_forests/port/javascript/inference/npm/dist/readme.txt

# Compile Typescript.
(
  cd yggdrasil_decision_forests/port/javascript/training
  npm install --save-dev webpack webpack-cli typescript ts-loader
  npx webpack
)

bazel build -c opt --config=wasm \
  //yggdrasil_decision_forests/port/javascript/training:create_release

mkdir -p dist
cp -f bazel-bin/yggdrasil_decision_forests/port/javascript/training/ydf.zip dist/ydf_training.zip
cp -f bazel-bin/yggdrasil_decision_forests/port/javascript/training/ydf.zip dist/javascript_wasm_inference.zip

# Extract library to NPM location
unzip dist/ydf_training.zip -d yggdrasil_decision_forests/port/javascript/training/npm/dist

# Remove the unnecessary readme.txt in the subfolder.
rm yggdrasil_decision_forests/port/javascript/training/npm/dist/readme.txt
