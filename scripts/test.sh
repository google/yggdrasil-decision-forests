#!/bin/bash
#
# This script runs all the Yggdrasil Decision Forests tests.

set -e

# Default Bazel command
BAZEL=${BAZEL:-bazel}

# Default Bazel options
# For maximum performance, uncomment the following line:
# BAZEL_OPTS="--copt=-mfma --copt=-mavx2 --copt=-mavx"
BAZEL_OPTS=${BAZEL_OPTS:-"--copt=-mavx2"}

echo "Running all tests..."
$BAZEL test $BAZEL_OPTS --test_output=errors //yggdrasil_decision_forests/...