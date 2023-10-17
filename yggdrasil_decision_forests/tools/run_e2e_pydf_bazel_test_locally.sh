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
#   third_party/yggdrasil_decision_forests/tools/run_e2e_pydf_bazel_test_locally.sh

set -ex

LOCAL_DIR="/usr/local/google/home/${USER}/git/yggdrasil-decision-forests"
CL=$(hg exportedcl)
echo "Current CL: ${CL}"
echo "Make sure the CL is synced!"

function run_export () {
  COPYBARA="/google/bin/releases/copybara/public/copybara/copybara"
  bazel test third_party/yggdrasil_decision_forests:copybara_test

  rm -fr "${LOCAL_DIR}"
  "${COPYBARA}" third_party/yggdrasil_decision_forests/copy.bara.sky presubmit_piper_to_gerrit "${CL}" \
    --dry-run --init-history --squash --force \
    --git-destination-path "${LOCAL_DIR}" --ignore-noop

  /google/bin/releases/opensource/thirdparty/cross/cross "${LOCAL_DIR}"
}

run_test() {
  cd "${LOCAL_DIR}/yggdrasil_decision_forests/port/python"

  DOCKER_IMAGE=gcr.io/tfx-oss-public/manylinux2014-bazel:bazel-5.3.0
  DOCKER_CONTAINER=compile_pydf
  YDF_PATH=$(realpath $PWD/../../..)
  YDF_DIRNAME=${YDF_PATH##*/}

  echo "Available containers:"
  sudo sudo docker container ls -a --size

  set +e  # Ignore error if the container already exist
  CREATE_DOCKER_FLAGS="-i -t -v ${PWD}/../../../../:/working_dir -w /working_dir/${YDF_DIRNAME}/yggdrasil_decision_forests/port/python"
  sudo docker create ${CREATE_DOCKER_FLAGS} --name ${DOCKER_CONTAINER} ${DOCKER_IMAGE}
  sudo docker start ${DOCKER_CONTAINER}
  set -e

  CMD='PYTHON=python3.9;$PYTHON -m venv /tmp/venv_$PYTHON;source /tmp/venv_$PYTHON/bin/activate;COMPILERS="gcc" ./tools/test_pydf.sh;$SHELL'
  sudo docker exec -it ${DOCKER_CONTAINER} /bin/bash -c "${CMD}"
}

run_export
run_test
