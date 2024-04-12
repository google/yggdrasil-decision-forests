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
#   third_party/yggdrasil_decision_forests/tools/run_e2e_pydf_test.sh

set -ex

LOCAL_DIR="/usr/local/google/home/${USER}/git/yggdrasil-decision-forests"
CL=$(hg exportedcl)
echo "Current CL: ${CL}"
echo "Make sure the CL is synced!"

function run_export () {
  COPYBARA="/google/bin/releases/copybara/public/copybara/copybara"
  bazel test third_party/yggdrasil_decision_forests:copybara_test

  sudo rm -fr "${LOCAL_DIR}"

  "${COPYBARA}" third_party/yggdrasil_decision_forests/copy.bara.sky presubmit_piper_to_gerrit "${CL}" \
    --dry-run --init-history --squash --force \
    --git-destination-path "${LOCAL_DIR}" --ignore-noop

  /google/bin/releases/opensource/thirdparty/cross/cross "${LOCAL_DIR}"
}

run_test() {
  cd "${LOCAL_DIR}/yggdrasil_decision_forests/port/python"

  DOCKER_IMAGE=quay.io/pypa/manylinux2014_x86_64@sha256:2e37241d9c9fbbccea009e59505a1384f9501a7bfea77b21fdcbf332c7036e70
  DOCKER_CONTAINER=compile_pydf_v2
  YDF_PATH=$(realpath $PWD/../../..)
  YDF_DIRNAME=${YDF_PATH##*/}

  echo "Available containers:"
  sudo sudo docker container ls -a --size

  set +e  # Ignore error if the container already exist
  CREATE_DOCKER_FLAGS="-i -t -p 8889:8889 --network host -v ${PWD}/../../../../:/working_dir -w /working_dir/${YDF_DIRNAME}/yggdrasil_decision_forests/port/python"
  sudo docker create ${CREATE_DOCKER_FLAGS} --name ${DOCKER_CONTAINER} ${DOCKER_IMAGE}
  sudo docker start ${DOCKER_CONTAINER}
  set -e

  CMD='yum update;yum install -y rsync;curl -L -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64;chmod +x /usr/local/bin/bazel;PYTHON=python3.11;$PYTHON -m venv /tmp/venv_$PYTHON;source /tmp/venv_$PYTHON/bin/activate;export COMPILERS="gcc";./tools/test_pydf.sh;./tools/build_pydf.sh python;$SHELL'

  # Only get a shell, uncomment the following line.
  # CMD='$SHELL'

  # If the test fails, you can restart it with:
  # ./tools/test_pydf.sh
  #
  # To build a pip package, run:
  # ./tools/build_pydf.sh python
  #
  # To start a notebook instance, run:
  # ./tools/start_notebook.sh
  #
  # To clear all the docker containers run:
  # sudo sudo docker container ls
  # sudo sudo docker stop [ID]
  # sudo docker system prune -a

  sudo docker exec -it ${DOCKER_CONTAINER} /bin/bash -c "${CMD}"
}

run_export
run_test
