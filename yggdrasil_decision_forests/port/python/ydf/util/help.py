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

def loading_data():

  class Help:

    def __str__(self) -> str:

      extra_in_files = ""
      extra_utilities = ""

      return f"""\
YDF supports the following dataset formats.

In-Memory

- Dict: {{column_name: column_data}}, where column_data can be lists of basic types or NumPy arrays (1D or 2D arrays for multi-dimensional columns).
- Pandas DataFrame: No support for multi-dimensional columns.
- Polars DataFrame.
- Xarray Dataset.
- ydf.VerticalDataset: For repeated use; create with `ydf.create_vertical_dataset`.
- Batched TensorFlow Dataset.
- PyGrain DataLoader/Dataset (Experimental, Linux only).

File-Based

Use typed paths (e.g., "format:/path") which can be sharded (e.g.`format:/path@n`) or use globs (e.g. `format:/path*`). Lists of paths are also supported. Use the prefix "gs://" for files hosted on Google Cloud Storage e.g. "csv:gs://my_bucket/dataset*".

- `csv:`: Small datasets. No multi-dimensional columns.
- `avro:`: Large datasets. Supports multi-dimensional columns.
- `tfrecord:`: Compressed TensorFlow Record. Note: TFRecord are not RecordIOs.
- `tfrecord-nocompression:`: Uncompressed TensorFlow Record.
{extra_in_files}

See `ydf.util.*` contains utilities for utilities to import datasets in memory:
- `ydf.utils.read_tf_record`: Read TF Records.
- `ydf.utils.write_tf_record`: Write TF Records.
{extra_utilities}

Googlers: See go/ydf/in_google for additional internal-only formats.
"""

    def _repr_html_(self) -> str:
      extra_in_files = ""
      extra_utilities = ""

      return f"""\
<p>YDF supports the following dataset formats.</p>

<p><b>In-Memory</b></p>
<ul>
  <li><b>Dict:</b> <code>{{column_name: column_data}}</code>, where <code>column_data</code> can be lists of basic types or NumPy arrays (1D or 2D arrays for multi-dimensional columns).</li>
  <li><b>Pandas DataFrame:</b> No support for multi-dimensional columns.</li>
  <li><b>Polars DataFrame</b>.</li>
  <li><b>Xarray Dataset</b>.</li>
  <li><b><code>ydf.VerticalDataset</code>:</b> For repeated use; create with <code>ydf.create_vertical_dataset</code>.</li>
  <li><b>Batched TensorFlow Dataset</b>.</li>
  <li><b>PyGrain DataLoader/Dataset:</b> (Experimental, Linux only).</li>
</ul>

<p><b>File-Based</b></p>

<p>Use typed paths (e.g., <code>"format:/path"</code>) which can be sharded (e.g., <code>format:/path@n</code>) or use globs (e.g., <code>format:/path*</code>). Lists of paths are also supported. Use the prefix <code>gs://</code> for files hosted on Google Cloud Storage e.g. <code>csv:gs://my_bucket/dataset*</code>.</p>

<ul>
  <li><b><code>csv:</code></b> Small datasets. No multi-dimensional columns.</li>
  <li><b><code>avro:</code></b> Large datasets. Supports multi-dimensional columns.</li>
  <li><b><code>tfrecord:</code></b> Compressed TensorFlow Record. <i>Note:</i> TFRecord are not RecordIOs.</li>
  <li><b><code>tfrecord-nocompression:</code></b> Uncompressed TensorFlow Record.</li>
  {extra_in_files}
</ul>

<p>See <code>ydf.util.*</code> for utilities to import datasets in memory:</p>
<ul>
<li><b><code>ydf.utils.read_tf_record:</code></b> Read TF Records.</li>
<li><b><code>ydf.utils.write_tf_record:</code></b> Write TF Records.</li>
{extra_utilities}
</ul>

<p><b>Googlers:</b> See <a href="http://go/ydf/in_google">go/ydf/in_google</a> for additional internal-only formats.</p>
"""

  return Help()
