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

import {
  EmbindString,
  InputFeature,
  InternalLearner,
  InternalModel,
  MainModule,
  vectorFloat,
  vectorString,
} from '../training_for_types';
import * as conversion from '../util/conversion';

declare var Module: MainModule;

/**
 * Creates a C++-compatible dataset wrapper from a dictionary of values.
 * To prevent memory leaks, the returned object must be freed manually.
 *
 * @param data The data to create the dataset from. The keys are the column names
 *     and the values are the column values.
 * @param labelKey The key of the column that contains the labels. This column
 *     may be processed differently.
 * @returns The C++ dataset.
 */
export function createDataset(
  data: {[key: string]: unknown[]},
  labelKey: string | null,
) {
  const vds = new Module.Dataset();
  try {
    const keys = Object.keys(data);
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      const values = data[key];
      if (Array.isArray(values)) {
        if (typeof values[0] === 'number') {
          const ccVector = conversion.numberArrayToVectorFloat(
            values as number[],
          );
          vds.addNumericalColumn(key, ccVector);
          ccVector.delete();
        } else if (typeof values[0] === 'string') {
          const ccVector = conversion.stringArrayToVectorString(
            values as string[],
          );
          vds.addCategoricalColumn(key, ccVector, key === labelKey);
          ccVector.delete();
        } else {
          throw new Error(`Unsupported column type for key ${key}`);
        }
      } else {
        throw new Error(`Unsupported column type for key ${key}`);
      }
    }
    return vds;
  } catch (e) {
    vds.delete();
    throw e;
  }
}
