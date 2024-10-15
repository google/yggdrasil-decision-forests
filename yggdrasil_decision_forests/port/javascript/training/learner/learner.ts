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
 * @fileoverview JS Wrapper for the learner classes.
 */

import {
  InternalLearner,
  MainModule,
} from '../training_for_types';

import * as dataset from '../dataset/dataset';
import * as model from '../model/model';

declare interface GenericLearnerInterface {
  train: (data: string) => void;
}
declare var Module: MainModule;
const FS: model.EmscriptenFS = (Module as any)['FS'];

abstract class GenericLearner implements GenericLearnerInterface {
  constructor(
    readonly learnerKey: string,
    readonly label: string,
    readonly task: string,
  ) {}

  train(data: string | {[key: string]: unknown[]}): model.Model {
    const internalLearner = new Module.InternalLearner();
    try {
      internalLearner.init(this.learnerKey, this.label, this.task);

      if (typeof data === 'string') {
        const fileName = `/train_${Math.floor(Math.random() * 0xffffffff)}.csv`;
        FS.writeFile(fileName, data);

        const startTime = performance.now();
        const internalModel = internalLearner.trainFromPath(`csv:${fileName}`);
        const endTime = performance.now();
        console.log(`Model trained in ${(endTime - startTime) / 1000}`);
        FS.unlink(fileName);
        return new model.Model(internalModel);
      } else if (typeof data === 'object') {
        return this.trainFromObject(data, internalLearner);
      } else {
        throw new Error(
          `Unknown data type ${typeof data}. Data must be either a string containing CSV data or a named dictionary of values.`,
        );
      }
    } finally {
      internalLearner.delete();
    }
  }

  private trainFromObject(
    data: {[key: string]: unknown[]},
    internalLearner: InternalLearner,
  ) {
    const vds = dataset.createDataset(data, this.label);
    try {
      const internalModel = internalLearner.trainFromDataset(vds);
      return new model.Model(internalModel);
    } finally {
      vds.delete();
    }
  }
}

/**
 * A wrapper class for the RandomForestLearner C++ class.
 *
 * This class is used to train a RandomForestModel in JavaScript. It is a wrapper
 * around the C++ class, and it exposes the necessary methods to the JavaScript
 * code.
 *
 * TODO: Add usage example.
 */
export class RandomForestLearner extends GenericLearner {
  constructor(label: string, task: string) {
    super('RANDOM_FOREST', label, task);
  }
}

/**
 * A wrapper class for the GradientBoostedTreesLearner C++ class.
 *
 * This class is used to train a GradientBoostedTreesModel in JavaScript. It is
 * a wrapperaround the C++ class, and it exposes the necessary methods to the
 * JavaScript code.
 *
 * TODO: Add usage example.
 */
export class GradientBoostedTreesLearner extends GenericLearner {
  constructor(label: string, task: string) {
    super('GRADIENT_BOOSTED_TREES', label, task);
  }
}

/**
 * A wrapper class for the CartLearner C++ class.
 *
 * This class is used to train a CartModel in JavaScript. It is a wrapper
 * around the C++ class, and it exposes the necessary methods to the JavaScript
 * code.
 *
 * TODO: Add usage example.
 */
export class CartLearner extends GenericLearner {
  constructor(label: string, task: string) {
    super('CART', label, task);
  }
}
