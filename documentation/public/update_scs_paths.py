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

"""Update path to assets hosted on SCS.

For example:
  ../../non-github-assets/ydf_blog_header.png
Will become:
  https://www.gstatic.com/ydf-docs-assets/ydf_blog_header.png
"""

import mkdocs_gen_files


def process_md_file(path):
  with open("documentation/public/docs/" + path, "r", encoding="utf-8") as f_in:
    with mkdocs_gen_files.open(path, "w") as f_out:
      data = f_in.read()
      data = data.replace(
          "../../non-github-assets/",
          "https://www.gstatic.com/ydf-docs-assets/",
      )
      f_out.write(data)


process_md_file("blog/posts/1_how_ml_models_generalize.md")
