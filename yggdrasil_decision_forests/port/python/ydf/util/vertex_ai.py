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

"""Utilities to use YDF with Google Vertex AI."""

import dataclasses
import json
import os
from typing import List, Union


@dataclasses.dataclass
class VertexAIClusterSpec:
  """Description of the cluster for Vertex AI.

  Attributes:
    workers: List of IP addresses of the worker nodes.
    is_worker: True if the current node is a worker node. False if it is the
      manager.
    port: Port number to use for communication between nodes.
  """

  workers: List[str]
  is_worker: bool
  port: int


def get_vertex_ai_cluster_spec(
    cluster_spec: Union[str, None] = None,
) -> VertexAIClusterSpec:
  """Parses the Vertex AI cluster specification.

  The cluster specification is a JSON string describing the nodes of the
  cluster. This specification is typically provided by Vertex AI as an
  environment variable.

  Usage example:

  In a train.py file runing in a Docker in Vertex AI.

  ```python
  # Gather the manager and workers configuration.
  cluster_config = get_vertex_ai_cluster_spec()
  print("cluster_config:", cluster_config)

  if cluster_config.is_worker:
    # This machine is running a worker.
    ydf.start_worker(cluster_config.port)
    return

  print("Train model with distribution")
  learner = ydf.DistributedGradientBoostedTreesLearner(
      label=...,
      working_dir=...,
      workers=cluster_config.workers,
      resume_training=True,
  )
  model = learner.train(args.train_ds)
  ```

  Args:
    cluster_spec: Cluster specification as a JSON string. If None, the
      `CLUSTER_SPEC` environment variable is used instead.

  Returns:
    The parsed cluster specification.
  """

  if cluster_spec is None:
    print("Get Vertex AI Cluster Spec from env. variable")
    cluster_spec = os.environ.get("CLUSTER_SPEC")
  json_cluster_spec = json.loads(cluster_spec)
  json_cluster = json_cluster_spec["cluster"]
  if list(json_cluster.keys()) != ["workerpool0", "workerpool1"]:
    raise ValueError(
        "Expecting two workpools: workerpool0 and workerpool1. Instead got:"
        f" {list(json_cluster.keys())!r}"
    )

  is_worker = json_cluster_spec["task"]["type"] == "workerpool1"
  my_ip = json_cluster[json_cluster_spec["task"]["type"]][
      json_cluster_spec["task"]["index"]
  ]
  port = int(my_ip.rsplit(":", 1)[1])
  return VertexAIClusterSpec(
      workers=json_cluster["workerpool1"],
      is_worker=is_worker,
      port=port,
  )
