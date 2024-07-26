# 2024.07.18-1
# See: https://quay.io/repository/pypa/manylinux2014_x86_64?tab=tags
FROM quay.io/pypa/manylinux2014_x86_64@sha256:025ab0fe69f5397510920afa23f26bf6e61955ed5d3868f09ec59996fba41b0b

WORKDIR /build_tools

ENV PATH="$PATH:/build_tools"
ADD https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 bazel
RUN chmod +x bazel

ENV CC=gcc

ENTRYPOINT ["/bin/bash", "-c"]
