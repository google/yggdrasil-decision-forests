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

"""Copy files from YDF repo into the doc."""

import mkdocs_gen_files

with open("CHANGELOG.md", "r", encoding="utf-8") as f_in:
  with mkdocs_gen_files.open("changelog.md", "w") as f_out:
    f_out.write("See also the [PYDF changelogs](changelog_pydf.md).\n\n")
    f_out.write(f_in.read())

with open("LICENSE", "r", encoding="utf-8") as f_in:
  with mkdocs_gen_files.open("LICENSE", "w") as f_out:
    f_out.write(f_in.read())

with open(
    "yggdrasil_decision_forests/port/python/CHANGELOG.md", "r", encoding="utf-8"
) as f_in:
  with mkdocs_gen_files.open("changelog_pydf.md", "w") as f_out:
    f_out.write(f_in.read())
