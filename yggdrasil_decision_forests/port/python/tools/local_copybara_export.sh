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
# Export the project using copybara locally, and run the pydf tests.
#
# Usage example:
#   third_party/yggdrasil_decision_forests/port/python/tools/local_copybara_export.sh

set -vex

LOCAL_DIR="/usr/local/google/home/${USER}/git/yggdrasil-decision-forests"
CL=$(hg exportedcl)
echo "Current CL: ${CL}"
echo "Make sure the CL is synced!"

function local_export () {
  COPYBARA="/google/bin/releases/copybara/public/copybara/copybara"
  bazel test third_party/yggdrasil_decision_forests:copybara_test

  sudo rm -fr "${LOCAL_DIR}"

  "${COPYBARA}" third_party/yggdrasil_decision_forests/copy.bara.sky presubmit_piper_to_gerrit "${CL}" \
    --dry-run --init-history --squash --force \
    --git-destination-path "${LOCAL_DIR}" --ignore-noop

  /google/bin/releases/opensource/thirdparty/cross/cross "${LOCAL_DIR}"
}

build_and_test() {
  cd "${LOCAL_DIR}/yggdrasil_decision_forests/port/python"
  sudo tools/release_linux.sh
}

local_export
build_and_test
