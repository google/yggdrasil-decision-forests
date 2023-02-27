# Hyper-Parameter Sweep

## Introduction

In this project, we train, evaluate and measure the inference speed of a large
number of models trained with different hyper-parameters. The outcome of this
process is a table containing, for each candidate hyper-parameter configuration,
model statistics (e.g., average tree depth) and model metrics (e.g., quality and
inference speed).

**Note:** If you are looking to tune hyper-parameters to optimize common model
metrics (e.g., the model accuracy), use the hyper-parameter tuner
([example](https://github.com/google/yggdrasil-decision-forests/blob/main/examples/hyperparameter_tuning.sh))
instead. It is much simpler to use.

## Dependencies

This project is implemented in c++. The following dependencies are required for
Linux.

```shell
apt-get update
apt-get -y --no-install-recommends install \
  ca-certificates \
  build-essential \
  g++-10 \
  clang-10 \
  git \
  python3 \
  python3-pip

python3 -m pip install numpy

# Install Bazel
wget -O bazelisk https://github.com/bazelbuild/bazelisk/releases/download/v1.14.0/bazelisk-linux-amd64
chmod +x bazelisk
mv bazelisk bazel
```

**Note:** If you install bazel this way, you have to replace `bazel` by
`./bazel` in `run_local.sh`.

## Files

This project is structured as follow:

-   `prepare_data.py`: Converts the `hls4ml_HLF` dataset from arff to csv format
    and applies preprocessing on the data before training.
-   `run_local.sh`: Run the hyper-parameter sweep on your local machine using
    multi-threading.
-   `BUILD`: Bazel builds file configuration.
-   `manager_main.cc`: The main/entry point of the program. Run the
    hyper-parameter sweep and export the results to a JSON file.
-   `worker.cc`: The code responsible to train, evaluate and benchmark the
    models.
-   `optimizer.proto`: Protobuf messages used for communication between
    `manager_main` and `worker`.
-   `plot_results.ipynb`: A python notebook / colab to analyze and plot the JSON
    file exported by `manager_main`.

## How to run this project

This project can be run locally or in a distributed computation setup.

### Local execution

#### 1.

Convert the dataset to csv format using `prepare_data.py`.By default,
`prepare_data.py` is configured to run a binary classification on class `t` vs
the others.

**Note:** Before running `prepare_data.py`, update the path defined in the
"options" section.

#### 2.

**Note:** Before running `run_local.sh`, update the `PROJECT` field.

Run the sweep locally with:

```
./run_local.sh
```

This command compiles and runs a hyper-parameter sweep on a small set of
hyper-parameters and exports the results to a CSV file. You can then use the ``
colab to plot the results.

**Note:** Look at the content of `run_local.sh` for an example of the parameters
of the sweep. For example, `--num_repetitions=2` indicates that each model has
trained twice the measure the variance. Check the `manager_main` documentation
(`manager_main --help`) or look at the`ABSL_FLAG`section in`manager_main.cc` for
an explanation of the available flags.

**Note:** If you have issues compiling the program directly, you can compile it
in a docker

### 3.

The results are available in the .json file in the `work_dir` directory defined
in `./run_local.sh`. For example, the first lines of the file will be something
like this (without the explanations):

```json
[
  {
# The name of the learning algorithm.
"algorithm": "GRADIENT_BOOSTED_TREES",
# The number of trees configured to train the model. Note that the effective
# number of trees can be smaller (e.g. if early stopping hits).
"num_trees": 5,
# An hyper-parameter. Check https://ydf.readthedocs.io/en/latest/hyper_parameters.html
# for the list of hyper-parameters.
"shrinkage": 0.1,
# An hyper-parameter.
"subsample": 0.1,
# An hyper-parameter.
"use_hessian_gain": true,
# An hyper-parameter.
"global_growth": true,
# An hyper-parameter. If max_depth=-1, there are no limit to the maximum depth.
"max_depth": -1,
# An hyper-parameter.
"max_nodes": 32, If max_nodes=-1, there are no limit to the maximum number of nodes.
# Unique index of the run.
"run_idx": 0,
# The number of trees in the model.
"effective_num_trees": 5,
# The number of nodes in the model.
"effective_num_nodes": 315,
# The accuracy of the model.
"accuracy": 0.860722,
# The AUCs of the model. The first AUC is always zero and can be ignored.
# In the case of multi-class classification, there AUCs are the one-vs-other
# AUCs. In the case of binary-classification, the second and third AUCs are
# always equal to each other.
"aucs": [0,0.944307,0.944307],
# Average model inference per example expressed in seconds.
"time_per_predictions_s": 7.8e-08,
# Index of the repetition.
"repetition_idx": 0
},
...
```

Plot the results with `plot_results.ipynb`.

## Distributed execution

To run this project at scale, the execution should be distributed.

Distributed execution is similar to local execution with some exceptions.
Notably, you need to:

-   Start workers running the `worker` binaries before starting `manager_main`.
-   The socket address of the worker should be provided to `manager_main` using
    the `distribute_config` flag.

### Start the workers

Each worker machine should execute the `worker` binary with a `--port` argument.
This port argument is the socket port used for communication between the chief
and workers. For example:

-   Machine #0: run `worker --port=2000`
-   Machine #1: run `worker --port=2000`
-   Machine #2: run `worker --port=2000`
-   Machine #3: run `worker --port=2000`

Each worker will train `--parallel_execution_per_worker` models in parallel. The
default value is 1. Training multiple models in parallel on a worker is more
efficient. However, it can also bias the inference speed estimation results.

### Start the manager

The chief/manager (`manager_main` binary) should be executed on a single
machine. The flag `--distribute_config` is used to specify the socket address of
each of the workers. For example:

```
--distribute_config=
    implementation_key:"GRPC"
    [yggdrasil_decision_forests.distribute.proto.grpc] {
      grpc_addresses {
        addresses: "<ip address of worker machine #0>:2000"
        addresses: "<ip address of worker machine #1>:2000"
        addresses: "<ip address of worker machine #2>:2000"
        addresses: "<ip address of worker machine #3>:2000"
      }
    }
```

**Note:** The configuration of `distribute_config` is similar to the ``
configuration for Yggdrasil Decision Forest Distributed training
([documentation](https://ydf.readthedocs.io/en/latest/cli_user_manual.html#distributed-training)).
