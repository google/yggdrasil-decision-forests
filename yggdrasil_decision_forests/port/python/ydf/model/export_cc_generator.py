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

"""Generates the c++ code to run a model for a unit-test."""


from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging

import ydf

_INPUT_MODEL = flags.DEFINE_string("input_model", None, "Path to input model")

_OUTPUT_CODE = flags.DEFINE_string(
    "output_code", None, "Path to generated c++ file."
)


def process(input_model: str, output_code: str) -> None:
  logging.info(
      "Loading model %s and generating cc code in %s", input_model, output_code
  )

  model = ydf.load_model(input_model)
  with open(output_code, "w") as f:
    f.write(model.to_cpp("123"))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  process(_INPUT_MODEL.value, _OUTPUT_CODE.value)


if __name__ == "__main__":
  app.run(main)
