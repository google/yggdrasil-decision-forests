/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Extra configs for the Karma test runner.
 *
 * @param {!Object} config
 */
module.exports = function(config) {
  basePath = 'third_party/yggdrasil_decision_forests/port/javascript/';
  config.files.push({
    pattern: basePath + 'inference_wasm/inference.js',
    watched: false,
    served: true,
    nocache: false,
    included: true,
  });
  config.files.push({
    pattern: basePath + 'inference_wasm/inference.wasm',
    watched: false,
    served: true,
    nocache: false,
    included: false,
  });
  config.files.push({
    pattern: basePath + 'example/model.zip',
    watched: false,
    served: true,
    nocache: false,
    included: false,
  });
  config.files.push({
    pattern: basePath + 'test_data/model_2.zip',
    watched: false,
    served: true,
    nocache: false,
    included: false,
  });
  config.files.push({
    pattern: basePath + 'test_data/model_3.zip',
    watched: false,
    served: true,
    nocache: false,
    included: false,
  });
  config.files.push({
    pattern: basePath + 'test_data/model_small_sst.zip',
    watched: false,
    served: true,
    nocache: false,
    included: false,
  });
  config.files.push({
    pattern: 'third_party/javascript/node_modules/jszip/jszip.min.js',
    watched: false,
    served: true,
    nocache: false,
    included: true,
  });
};