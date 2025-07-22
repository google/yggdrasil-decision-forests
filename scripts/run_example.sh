#!/bin/bash
#
# This script builds and runs the main beginner example.

set -e

# Default Bazel command
BAZEL=${BAZEL:-bazel}

# Default Bazel options
# For maximum performance, uncomment the following line:
# BAZEL_OPTS="--copt=-mfma --copt=-mavx2 --copt=-mavx"
BAZEL_OPTS=${BAZEL_OPTS:-"--copt=-mavx2"}

echo "Building and running the beginner example..."
$BAZEL run $BAZEL_OPTS //yggdrasil_decision_forests/examples:beginner
