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



# Converts a non-submitted Piper CL to a Bazel project in a local directory and
# run all the tests.
#
# Usage example:
#   third_party/yggdrasil_decision_forests/tools/local_copybara_export.sh
#   INTERACTIVE=1 third_party/yggdrasil_decision_forests/tools/local_copybara_export.sh

set -ex

LOCAL_DIR="/usr/local/google/home/${USER}/git/yggdrasil-decision-forests"

CL=$(hg exportedcl)
echo "Current CL: ${CL}"
echo "Make sure the CL is synced!"

function run_export () {
  COPYBARA="/google/bin/releases/copybara/public/copybara/copybara"
  bazel test third_party/yggdrasil_decision_forests:copybara_test

  rm -fr ${LOCAL_DIR}
  ${COPYBARA} third_party/yggdrasil_decision_forests/copy.bara.sky presubmit_piper_to_gerrit ${CL} \
    --dry-run --init-history --squash --force \
    --git-destination-path ${LOCAL_DIR} --ignore-noop

  /google/bin/releases/opensource/thirdparty/cross/cross ${LOCAL_DIR}
}

run_test() {
  cd "${LOCAL_DIR}"

  DOCKER_IMAGE=ubuntu:20.04
  DOCKER_CONTAINER=compile_ydf

  echo "Available containers:"
  sudo sudo docker container ls -a --size

  set +e  # Ignore error if the container already exist
  CREATE_DOCKER_FLAGS="-i -t -v ${PWD}/..:/working_dir -w /working_dir/yggdrasil-decision-forests"
  sudo docker create ${CREATE_DOCKER_FLAGS} --name ${DOCKER_CONTAINER} ${DOCKER_IMAGE}
  sudo docker start ${DOCKER_CONTAINER}
  set -e

  echo "YOu can run one of the following commands:"
  echo "INSTALL_DEPENDENCIES=1 COMPILERS="clang-12" CPP_VERSIONS="14" ./tools/test_bazel.sh"
  echo "INSTALL_DEPENDENCIES=1 COMPILERS="clang-12" CPP_VERSIONS="17" ./tools/test_bazel.sh"
  if [[ "$INTERACTIVE" = 1 ]]; then
    CMD='$SHELL'
  else
    CMD='INSTALL_DEPENDENCIES=1 COMPILERS="clang-12" CPP_VERSIONS="17" ./tools/test_bazel.sh;$SHELL'
  fi

  sudo docker exec -it ${DOCKER_CONTAINER} /bin/bash -c "${CMD}"
}

run_export
run_test

