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

RUN pip3 install numpy

ADD . $HOME

WORKDIR /home/developer/examples/model_server

# Compile the model server example
RUN bazel build --config=linux_cpp17 --config=linux_avx2 //:model_server_cc

# Run the model server example
CMD ["./bazel-bin/model_server_cc", "--dataset_dir=../../yggdrasil_decision_forests/test_data/dataset", "--output_dir=/tmp/yggdrasil_decision_forest"]