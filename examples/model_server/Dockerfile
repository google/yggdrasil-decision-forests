ARG UBUNTU_VERSION=22.04

FROM ubuntu:${UBUNTU_VERSION} as builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y \
        python3 python3-pip wget curl unzip git nano nvidia-cuda-dev libprotobuf-dev protobuf-compiler

# Set Home
WORKDIR /home/developer
ENV HOME /home/developer

# Install Bazel
ENV BAZEL_VERSION 5.1.1
RUN wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
RUN chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
RUN ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user
RUN cp -r $HOME/bin/bazel* /usr/bin/

ADD . $HOME

# begin hack
# get the files associated with the tensorflow target, but do not build yet
# issue: https://github.com/tensorflow/tensorflow/issues/56399
RUN bazel fetch @org_tensorflow//tensorflow
RUN sed -i -e "218s/result\.stderr or //" $(find /home/developer/.cache/bazel -path "*/org_tensorflow/third_party/remote_config/common.bzl")
# end hack

# Compile and run the model server example
WORKDIR ./examples/model_server
CMD ["./compile_and_run.sh"]