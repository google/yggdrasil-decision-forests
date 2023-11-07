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

"""Worker for distributed training."""

from typing import Callable, Optional

from ydf.cc import ydf as ydf_cc


def start_worker(
    port: int, blocking: bool = True
) -> Optional[Callable[[], None]]:
  """Starts a worker locally on the given port.

  The addresses of workers are passed to learners with the `workers` argument.

  Usage example:

  ```python
  # On worker machine #0 at address 192.168.0.1
  ydf.start_worker(9000)

  # On worker machine #1 at address 192.168.0.2
  ydf.start_worker(9000)

  # On manager
  learner = ydf.DistributedGradientBoostedTreesLearner(
        label = "my_label",
        working_dir = "/shared/working_dir,
        resume_training = True,
        workers = ["192.168.0.1:9000", "192.168.0.2:9000"],
    ).train(dataset)
  ```

  Example with non-blocking call:

  ```python
  # On worker machine
  stop_worker = start_worker(blocking=False)
  # Do some work with the worker
  stop_worker() # Stops the worker
  ```

  Args:
    port: TCP port of the worker.
    blocking: If true (default), the function is blocking until the worker is
      stopped (e.g., error, interruption by the manager). If false, the function
      is non-blocking and returns a callable that, when called, will stop the
      worker.

  Returns:
    Callable to stop the worker. Only returned if `blocking=True`.
  """

  if blocking:
    ydf_cc.StartWorkerBlocking(port)
    return None

  uid = ydf_cc.StartWorkerNonBlocking(port)
  return lambda: ydf_cc.StopWorkerNonBlocking(uid)
