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



# Starts a local https server hosting the static files (e.g. image, scripts,
# wasm) of the addon for development. Make sure to update the url in the
# "DataUrl" function accordingly.
#
# Usage example:
#
#   ./third_party/yggdrasil_decision_forests/port/javascript/tools/build_and_run_local_server.sh
#
set -vex

# Root dir of the https server.
ROOT_DIR="/tmp/ydf_in_js"

# Server port
PORT=2345

# Directory containing the project source code.
PROJECT_DIR=third_party/yggdrasil_decision_forests/port/javascript

# Build project
bazel build \
  --config=gce \
  --config=force_full_protos \
  -c opt \
  //${PROJECT_DIR}:create_release \
  //third_party/javascript/node_modules/jszip:jszip.min.js

mkdir -p ${ROOT_DIR}

# Create a ssl certificate.
if [[ ! -f "${ROOT_DIR}/cert.pem" ]]; then
    echo "Generate a certificate"
    openssl req -x509 -newkey rsa:2048 -keyout "${ROOT_DIR}/key.pem" -out "${ROOT_DIR}/cert.pem" -days 365 -nodes -batch
fi

# Copy the data to the https server.
unzip -o bazel-genfiles/${PROJECT_DIR}/ydf.zip -d /${ROOT_DIR}/ydf/
cp -f ${PROJECT_DIR}/example/example.html /${ROOT_DIR}/
cp -f ${PROJECT_DIR}/example/*.zip /${ROOT_DIR}/
cp -f bazel-bin/third_party/javascript/node_modules/jszip/jszip.min.js ${ROOT_DIR}/jszip.js

# List content of root dir
tree ${ROOT_DIR}

# Start the server.
python3 third_party/yggdrasil_decision_forests/tools/https_server.py ${ROOT_DIR} ${PORT}
