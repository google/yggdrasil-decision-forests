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

"""Python wrapper for the model metadata."""

import dataclasses
from typing import Dict, Optional, Union

from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.utils import log


@dataclasses.dataclass
class ModelMetadata:
  """Metadata information stored in the model.

  Attributes:
    owner: Owner of the model, defaults to empty string for the open-source
      build of YDF.
    created_date: Unix timestamp of the model training (in seconds).
    uid: Unique identifier of the model.
    framework: Framework used to create the model. Defaults to "Python YDF" for
      models trained with the Python API.
    custom_fields: Custom fields to be populated by the user.
  """

  owner: Optional[str] = None
  created_date: Optional[int] = None
  uid: Optional[int] = None
  framework: Optional[str] = None
  custom_fields: Dict[str, Union[bytes, str]] = dataclasses.field(
      default_factory=lambda: {}
  )

  def _to_proto_type(self) -> abstract_model_pb2.Metadata:
    proto_metadata = abstract_model_pb2.Metadata(
        owner=self.owner,
        created_date=self.created_date,
        uid=self.uid,
        framework=self.framework,
    )
    if self.custom_fields is not None:
      for key, value in self.custom_fields.items():
        if isinstance(value, str):
          value = value.encode()
        proto_metadata.custom_fields.append(
            abstract_model_pb2.Metadata.CustomField(key=key, value=value)
        )
    return proto_metadata

  @classmethod
  def _from_proto_type(cls, proto: abstract_model_pb2.Metadata):
    py_metadata = ModelMetadata(
        owner=proto.owner if proto.HasField("owner") else None,
        created_date=proto.created_date
        if proto.HasField("created_date")
        else None,
        uid=proto.uid if proto.HasField("uid") else None,
        framework=proto.framework if proto.HasField("framework") else None,
    )
    if proto.custom_fields:
      for field in proto.custom_fields:
        key = field.key
        if key in py_metadata.custom_fields:
          log.warning(
              f"This model contains duplicate key {key} in the custom field's"
              " of the model metadata. This is not supported. The custom"
              " fields of the model metadata may not have been fully loaded."
          )
        try:
          py_metadata.custom_fields[key] = field.value.decode("utf-8")
        except UnicodeDecodeError:
          py_metadata.custom_fields[key] = field.value
    return py_metadata
