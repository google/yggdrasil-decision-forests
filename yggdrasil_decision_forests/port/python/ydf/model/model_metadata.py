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
from typing import Optional

from yggdrasil_decision_forests.model import abstract_model_pb2


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
  """

  owner: Optional[str] = None
  created_date: Optional[int] = None
  uid: Optional[int] = None
  framework: Optional[str] = None

  def _to_proto_type(self) -> abstract_model_pb2.Metadata:
    return abstract_model_pb2.Metadata(
        owner=self.owner,
        created_date=self.created_date,
        uid=self.uid,
        framework=self.framework,
    )

  @classmethod
  def _from_proto_type(cls, proto: abstract_model_pb2.Metadata):
    return ModelMetadata(
        owner=proto.owner if proto.HasField("owner") else None,
        created_date=proto.created_date
        if proto.HasField("created_date")
        else None,
        uid=proto.uid if proto.HasField("uid") else None,
        framework=proto.framework if proto.HasField("framework") else None,
    )
