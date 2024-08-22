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

import importlib.util
import os
import sys
import tempfile
from typing import Any, Dict

from absl.testing import absltest
from absl.testing import parameterized
from fastapi import testclient

import ydf  # In the world, use "import ydf"
from ydf.model import export_docker
from ydf.model import model_lib
from ydf.utils import test_utils

# ydf alias for sub-modules
sys.modules["ydf"] = ydf


def load_module_by_src(name: str, src_path: str, work_dir: str):
  """Loads a python module from a source .py file."""
  spec = importlib.util.spec_from_file_location(name, src_path)
  assert spec is not None
  assert spec.loader is not None
  my_module = importlib.util.module_from_spec(spec)
  sys.modules[name] = my_module
  save_pwd = os.getcwd()
  os.chdir(work_dir)
  spec.loader.exec_module(my_module)
  os.chdir(save_pwd)
  return my_module


class ExportDocker(parameterized.TestCase):

  def test_gen_model_fields(self):
    model_dir = os.path.join(test_utils.ydf_test_data_path(), "model")
    model = model_lib.load_model(
        os.path.join(model_dir, "adult_binary_class_gbdt")
    )
    self.assertEqual(
        export_docker.gen_model_fields(model, "  ", {"age": "AGE"}),
        """\
  AGE: float = math.nan
  workclass: str = ""
  fnlwgt: float = math.nan
  education: str = ""
  education_num: str = ""
  marital_status: str = ""
  occupation: str = ""
  relationship: str = ""
  race: str = ""
  sex: str = ""
  capital_gain: float = math.nan
  capital_loss: float = math.nan
  hours_per_week: float = math.nan
  native_country: str = \"\"""",
    )

  @parameterized.parameters(
      (
          "adult_binary_class_gbdt",
          {
              "age": 20,
              "workclass": "something",
              "fnlwgt": 12.5,
          },
          {
              "label_classes": ["<=50K", ">50K"],
              "predictions": 0.0023047993890941143,
          },
      ),
      (
          "iris_multi_class_gbdt",
          {
              "Sepal.Length": 1.1,
              "Sepal.Width": 1.2,
          },
          {
              "label_classes": ["virginica", "versicolor", "setosa"],
              "predictions": [
                  2.9043911720094684e-10,
                  1.0,
                  2.9081884123094426e-10,
              ],
          },
      ),
      (
          "abalone_regression_gbdt",
          {
              "age": 20,
              "workclass": "something",
              "fnlwgt": 12.5,
          },
          {
              "predictions": 9.740714073181152,
          },
      ),
      (
          "sst_binary_class_gbdt",
          {
              "sentence": [
                  "uneasy",
                  "mishmash",
                  "of",
                  "styles",
                  "and",
                  "genres",
              ],
          },
          {
              "label_classes": ["1", "0"],
              "predictions": 0.4442742168903351,
          },
      ),
  )
  def test_to_docker(
      self,
      model_name: str,
      request: Dict[str, Any],
      expected_response: Dict[str, Any],
  ):
    model = model_lib.load_model(
        os.path.join(test_utils.ydf_test_data_path(), "model", model_name)
    )

    with tempfile.TemporaryDirectory() as tempdir:
      docker_dir = os.path.join(tempdir, f"end_point_{model_name}")
      model.to_docker(docker_dir)
      self.assertSetEqual(
          set(os.listdir(docker_dir)),
          set([
              "test_locally.sh",
              "main.py",
              "readme.txt",
              "Dockerfile",
              "model",
              "requirements.txt",
              "deploy_in_google_cloud.sh",
          ]),
      )

      endpoint_module = load_module_by_src(
          "main", os.path.join(docker_dir, "main.py"), docker_dir
      )
      client = testclient.TestClient(endpoint_module.app)
      response = client.post("/predict", json=request)
      self.assertEqual(response.json(), expected_response)
      self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
  absltest.main()
