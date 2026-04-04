# See: https://quay.io/repository/pypa/manylinux_2_28_x86_64?tab=tags
FROM quay.io/pypa/manylinux_2_28_x86_64@sha256:4d62ce9d1ca259171e7cde403686e538e6f8fdc517660a557a66e102577d67d3

WORKDIR /build_tools

ENV PATH="$PATH:/build_tools"
ADD https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64 bazel
RUN chmod +x bazel

ENV CC=gcc

ENTRYPOINT ["/bin/bash", "-c"]
