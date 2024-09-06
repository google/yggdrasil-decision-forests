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

r"""Benchmark the I/O speed.

Usage example:

  bazel run -c opt \
    //external/ydf_cc/yggdrasil_decision_forests/port/python/ydf/monitoring:benchmark_io_speed_main\
    -- --alsologtostderr
"""

from absl import app
from absl import flags

from ydf.monitoring import benchmark_io_speed as benchmark
from ydf.utils import log


_WORK_DIR = flags.DEFINE_string(
    "work_dir",
    "/tmp/benchmark_io_speed",
    "",
)
_TESTS = flags.DEFINE_multi_enum("tests", None, ["tfrecord", "recordio"], "")
_NUM_EXAMPLES = flags.DEFINE_integer("num_examples", 100_000, "")
_NUM_FEATURES = flags.DEFINE_integer("num_features", 20, "")
_NUM_SHARDS = flags.DEFINE_integer("num_shards", 100, "")
_CREATE_WORKDIR = flags.DEFINE_bool("create_workdir", True, "")


def main(argv):
  del argv
  log.info(
      "Results:\n%s",
      benchmark.run(
          work_dir=_WORK_DIR.value,
          tests=_TESTS.value,
          num_examples=_NUM_EXAMPLES.value,
          num_features=_NUM_FEATURES.value,
          num_shards=_NUM_SHARDS.value,
          create_workdir=_CREATE_WORKDIR.value,
      ),
  )


if __name__ == "__main__":
  app.run(main)
