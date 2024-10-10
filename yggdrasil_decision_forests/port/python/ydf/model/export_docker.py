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

"""Utility to export the model to a docker end-point."""

import os
import re
from typing import Dict, Sequence
from ydf.dataset import dataspec as dataspec_lib
from ydf.model import generic_model


def to_docker(
    model: generic_model.GenericModel, path: str, exist_ok: bool
) -> None:
  """See `model.to_docker` for the documentation of this function."""

  os.makedirs(path, exist_ok=exist_ok)
  model.save(os.path.join(path, "model"))

  with open(os.path.join(path, "deploy_in_google_cloud.sh"), "w") as f:
    f.write(content_deploy_sh())

  with open(os.path.join(path, "readme.txt"), "w") as f:
    f.write(content_readme())

  with open(os.path.join(path, "test_locally.sh"), "w") as f:
    f.write(content_test_sh())

  with open(os.path.join(path, "main.py"), "w") as f:
    f.write(content_main_py(model))

  with open(os.path.join(path, "Dockerfile"), "w") as f:
    f.write(content_dockerfile())

  with open(os.path.join(path, "requirements.txt"), "w") as f:
    f.write(content_requirements())


def content_readme():
  return """\
This directory contains an automatically generated Docker endpoint for running a YDF Machine Learning model.

- test_locally.sh: Show how to run the endpoint on your local machine.
- deploy_in_google_cloud.sh: Deploy the model to Google Cloud.
- For a comprehensive walkthrough, follow the tutorial at https://ydf.readthedocs.io/en/latest/tutorial/to_docker/.
"""


def content_deploy_sh():
  return """\
#!/bin/bash
set -vex

# Pack the model into a Docker endpoint, upload it to Google Cloud, and start a service endpoint.
# Look at the logs for the end-point url.
# Monitor your end-points at: https://pantheon.corp.google.com/run
gcloud run deploy ydf-predict --source . --region us-east1
"""


def content_test_sh():
  return """\
#!/bin/bash
set -vex

# Pack the model into a Docker endpoint
docker build --platform linux/amd64 -t ydf_predict_image .

# Start the docker endpoint locally
CONTAINER_NAME=$(docker run --rm -p 8080:8080 -d ydf_predict_image)

# Wait for the endpoint to load
sleep 5

# Send a prediction request to the endpoint
curl -X 'POST' \
  'localhost:8080/predict' \
  -H "accept: application/json" \
  -H 'Content-Type: application/json' \
  -d '{}'

# Stop the endpoint
docker stop ${CONTAINER_NAME}
"""


def gen_model_fields(
    model: generic_model.GenericModel,
    prefix: str,
    feature_to_field: Dict[str, str],
) -> str:
  """Generates the pydantic string definition of the model features.

  Args:
    model: Model.
    prefix: String to append to each line.
    feature_to_field: If a feature name is a key in this dictionary, use the
      corresponding dictionary value instead of the feature name.

  Returns:
    pydantic string definition.
  """
  list_fields = []
  for feature in model.input_features():
    if (
        feature.semantic == dataspec_lib.Semantic.NUMERICAL
        or feature.semantic == dataspec_lib.Semantic.DISCRETIZED_NUMERICAL
    ):
      # Note: int are implicitely converted to float.
      type_value = "float"
      default_value = "math.nan"
    elif feature.semantic == dataspec_lib.Semantic.CATEGORICAL:
      type_value = "str"
      default_value = '""'
    elif feature.semantic == dataspec_lib.Semantic.CATEGORICAL_SET:
      type_value = "List[str]"
      default_value = None  # catset values cannot be missing
    elif feature.semantic == dataspec_lib.Semantic.BOOLEAN:
      type_value = "bool"
      default_value = None  # boolean values cannot be missing
    else:
      raise ValueError(
          f"Non supported semantic {feature.semantic} for feature"
          f" {feature.name}"
      )
    default_value_def = (
        f" = {default_value}" if default_value is not None else ""
    )
    field_name = feature_to_field.get(feature.name, feature.name)
    list_fields.append(f"{prefix}{field_name}: {type_value}{default_value_def}")

  return "\n".join(list_fields)


def feature_name_to_field_name(feature_name: str) -> str:
  """Normalizes a feature name into a valid python field name."""
  feature_name = re.sub(r"[^a-zA-Z0-9_]", "_", feature_name)
  if feature_name[0].isdigit():
    feature_name = "_" + feature_name
  feature_name = re.sub(r"_+", "_", feature_name)
  return feature_name


def feature_name_to_field_name_dict(
    feature_names: Sequence[str],
) -> Dict[str, str]:
  """Creates a mapping of feature names to valid python field names."""
  output = {}
  for feature_name in feature_names:
    field_name = feature_name_to_field_name(feature_name)
    if field_name != feature_name:
      output[feature_name] = field_name
  return output


def content_main_py(model: generic_model.GenericModel) -> str:
  """Generates the content of the main.py file."""

  pydantic_module = "pydantic"

  feature_to_field = feature_name_to_field_name_dict(
      model.input_feature_names()
  )
  model_fields = gen_model_fields(model, "  ", feature_to_field)

  if model.task() == generic_model.Task.CLASSIFICATION:
    prediction_dtype_value = (
        "float" if len(model.label_classes()) == 2 else "List[float]"
    )
    initialize = """\
label_classes = model.label_classes()
"""
    output_fields = f"""\
  predictions: {prediction_dtype_value}
  label_classes: List[str]
"""
    return_def = """\
    predictions=prediction_batch[0],
    label_classes=label_classes,
"""
  else:
    initialize = ""
    output_fields = """\
  predictions: float
"""
    return_def = """\
    predictions=prediction_batch[0],
"""

  if not feature_to_field:
    maybe_fix_feature_names = ""
  else:
    field_to_feature = {v: k for k, v in feature_to_field.items()}
    initialize += f"""
# Mapping between the name of the model features and the attributes of the
# Example class. This is necessary because the feature names are not valid
# python attribute names. Train a model with feature names that look like python
# variables to remove this block.
field_to_feature = {field_to_feature!r}
"""

    maybe_fix_feature_names = """\
  example_batch = { field_to_feature.get(k,k):v for k,v in example_batch.items()}"""

  return f"""\
from typing import Any, Dict, List
from fastapi import FastAPI
from {pydantic_module} import BaseModel
import ydf
import math

app = FastAPI()

model = ydf.load_model("model")
{initialize}

class Example(BaseModel):
{model_fields}


class Output(BaseModel):
{output_fields}


@app.post("/predict")
async def predict(example: Example) -> Output:
  # Wrap the example features into a batch i.e., a list. If multiple examples
  # are available at the same time, it is more efficient to group them and run a
  # single prediction with "predict_batch".
  example_batch: Dict[str, List[Any]] = {{
      k: [v] for k, v in example.dict().items()
  }}
{maybe_fix_feature_names}
  prediction_batch = model.predict(example_batch).tolist()

  return Output(
{return_def}
  )


@app.post("/predict_batch")
async def predict_batch(example_batch):
  return model.predict(example_batch).tolist()

"""


def content_dockerfile() -> str:
  return """\
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY main.py .
COPY model model
EXPOSE 8080
CMD ["fastapi", "run", "main.py", "--port", "8080"]
"""


def content_requirements() -> str:
  return """\
fastapi[all]
pydantic
ydf
"""
