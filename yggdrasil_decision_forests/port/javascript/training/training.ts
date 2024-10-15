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

import {MainModule} from './training_for_types';

import {
  CartLearner,
  GradientBoostedTreesLearner,
  RandomForestLearner,
} from './learner/learner';
import {loadModelFromZipBlob} from './model/model';

declare var Module: MainModule;

export {
  CartLearner,
  GradientBoostedTreesLearner,
  loadModelFromZipBlob,
  RandomForestLearner,
};

(Module as any)['RandomForestLearner'] = RandomForestLearner;
(Module as any)['GradientBoostedTreesLearner'] = GradientBoostedTreesLearner;
(Module as any)['CartLearner'] = CartLearner;
(Module as any)['loadModelFromZipBlob'] = loadModelFromZipBlob;
