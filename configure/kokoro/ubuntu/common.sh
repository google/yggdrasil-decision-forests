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


#
# Options
#  GCC_VERSION: What version of GCC to use e.g. "9"
#  RUN_TESTS: Run the unit tests e.g. 0 or 1.
#  EXPORT_BINARIES: Export the binaries to x20 e.g. 0 or 1.
#

set -e
set -x

BAZEL_VERSION=3.7.2

# Print Linux version details.
cat /etc/{os,lsb}-release

# Move to the root of the repo.
cd "${KOKORO_ARTIFACTS_DIR}/git/yggdrasil_decision_forests"

# Output directory that will contain:
# - Test logs
# - Binaries
mkdir -p output

# Export the test logs when the scrip ends.
capture_test_logs() {
  echo "Collecting test logs"

  DST="$KOKORO_ARTIFACTS_DIR/git/yggdrasil_decision_forests/output"

  # Note: Do not work if grouped.
  find -L bazel-testlogs -name "test.log" -exec cp --parents {} "$DST" \;
  find -L bazel-testlogs -name "test.xml" -exec cp --parents {} "$DST" \;

  # Remove the x flag from test.log files as it crashes Kokoro.
  chmod -R -x+X $DST

  # Converts the file with sponge extension.
  # Currently crashes the Kokoro pipeline.
  # find -L "$DST" -name "test.log" -exec rename 's/test.log/sponge_log.log/' {} \;
  # find -L "$DST" -name "test.xml" -exec rename 's/test.xml/sponge_log.xml/' {} \;
}
trap capture_test_logs EXIT

# Create an empty directory expected by Kokoro.
mkdir -p "${KOKORO_ARTIFACTS_DIR}/artifacts"
echo "nothing" > "${KOKORO_ARTIFACTS_DIR}/artifacts/toto.txt"

# Install GCC
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get -qq update
sudo apt-get -qq install -y gcc-${GCC_VERSION} g++-${GCC_VERSION}
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION}
sudo update-alternatives --set gcc "/usr/bin/gcc-${GCC_VERSION}"
gcc --version
g++ --version

# Install Bazel
use_bazel.sh ${BAZEL_VERSION}
BAZEL=bazel
which ${BAZEL}
${BAZEL} version

echo "====================================================="
echo "1. With TensorFlow IO, c++14 and without abslStatusOr"
echo "====================================================="
# The most likely way to fail.

FLAGS="--config=linux_cpp14 --config=linux_avx2 --config=use_tensorflow_io --define=no_absl_statusor=1"
bazel build //yggdrasil_decision_forests/cli/...:all ${FLAGS}

if [ "${RUN_TESTS}" = 1 ]; then
  bazel test //yggdrasil_decision_forests/{cli,dataset,learner,metric,model,serving}/...:all //examples:beginner_cc ${FLAGS}
fi

echo "====================================================="
echo "2. Without TensorFlow IO, c++17 and with abslStatusOr"
echo "====================================================="

FLAGS="--config=linux_cpp17 --config=linux_avx2"
bazel build //yggdrasil_decision_forests/cli/...:all ${FLAGS}

if [ "${RUN_TESTS}" = 1 ]; then
  bazel test //yggdrasil_decision_forests/{cli,metric,model,serving,utils}/...:all //examples:beginner_cc ${FLAGS}
fi

echo "=================================================="
echo "3. With TensorFlow IO, c++17 and with abslStatusOr"
echo "=================================================="
# The most complete way

FLAGS="--config=linux_cpp17 --config=linux_avx2 --config=use_tensorflow_io"
bazel build //yggdrasil_decision_forests/cli/...:all ${FLAGS}
if [ "${RUN_TESTS}" = 1 ]; then
  bazel test //yggdrasil_decision_forests/...:all ${FLAGS}
fi

# Export back the binary
if [ "${EXPORT_BINARIES}" = 1 ]; then
  cp bazel-bin/yggdrasil_decision_forests/cli/{show_dataspec,benchmark_inference,convert_dataset,evaluate,infer_dataspec,predict,show_dataspec,show_model,train} output/
  # Output files should not be executables.
  chmod a-x output/*
fi

# echo "Files before capturing logs"
# pwd
# ls -L -R -l bazel-testlogs
# ls -L -R -l output

capture_test_logs
trap - EXIT

# echo "Files after capturing logs"
# pwd
# ls -L -R -l output

echo "================================"
echo "End of common compilation script"
echo "================================"
