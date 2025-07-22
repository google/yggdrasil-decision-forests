#!/bin/bash
#
# This script builds the Yggdrasil Decision Forests C++ library and examples.

set -e

# Default Bazel command
BAZEL=${BAZEL:-bazel}

# Default Bazel options
# For maximum performance, uncomment the following line:
# BAZEL_OPTS="--copt=-mfma --copt=-mavx2 --copt=-mavx"
BAZEL_OPTS=${BAZEL_OPTS:-"--copt=-mavx2"}

echo "Building Yggdrasil Decision Forests..."
$BAZEL build $BAZEL_OPTS //yggdrasil_decision_forests/...