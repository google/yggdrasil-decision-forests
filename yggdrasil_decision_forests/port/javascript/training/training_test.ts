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

import 'jasmine';

import * as YDFTrainingwrapper from './training';

declare global {
  interface Window {
    YDFTraining: (options: unknown) => Promise<any>;
  }
}

function csvToDictionary(data: string): {
  [key: string]: Array<string | number>;
} {
  const rows = data.trim().split('\n');
  const headers = rows.shift()!.split(',');
  const result: {[key: string]: Array<string | number>} = {};

  headers.forEach((header) => {
    result[header.trim()] = [];
  });

  rows.forEach((row) => {
    const values = row.split(',');
    values.forEach((value, index) => {
      const header = headers[index].trim();
      const trimmedValue = value.trim();
      const numberValue = Number(trimmedValue);
      result[header].push(isNaN(numberValue) ? trimmedValue : numberValue);
    });
  });

  return result;
}

describe('YDF Training', () => {
  let YDFTraining: any | null = null;

  const adultTrainUrl =
    '/base/third_party/yggdrasil_decision_forests/test_data/dataset/adult_train.csv';
  const adultTestUrl =
    '/base/third_party/yggdrasil_decision_forests/test_data/dataset/adult_test.csv';
  const abaloneUrl =
    '/base/third_party/yggdrasil_decision_forests/test_data/dataset/abalone.csv';

  beforeAll(async () => {
    // Load library
    await window
      .YDFTraining({
        locateFile: (filename: string, dir: string) => {
          return dir + filename;
        },
      })
      .then((m) => {
        YDFTraining = m;
        console.log('The library is loaded');
      });
  });

  it('trains a model on the CSV Adult dataset', async () => {
    expect(YDFTraining).not.toBe(null);
    const learner: YDFTrainingwrapper.RandomForestLearner =
      new YDFTraining.RandomForestLearner('income', 'CLASSIFICATION');
    const adultTrain = await fetch(adultTrainUrl)
      .then((r) => r.blob())
      .then((b) => b.text());
    const adultTest = await fetch(adultTestUrl)
      .then((r) => r.blob())
      .then((b) => b.text());
    const model = learner.train(adultTrain);
    const predictions = model.predict(adultTest);
    expect(predictions).toHaveSize(9769);
    model.unload();
  });

  it('trains a RF model on an in-memory dataset', async () => {
    expect(YDFTraining).not.toBe(null);
    const learner: YDFTrainingwrapper.RandomForestLearner =
      new YDFTraining.RandomForestLearner('col_label', 'CLASSIFICATION');
    const data = {
      'col_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      'col_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
      'col_label': ['x', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'x'],
    };
    const model = learner.train(data);
    const predictions = model.predict(data);
    expect(predictions).toHaveSize(10);
    model.unload();
  });

  it('trains a GBT model on an in-memory dataset', async () => {
    expect(YDFTraining).not.toBe(null);
    const learner: YDFTrainingwrapper.GradientBoostedTreesLearner =
      new YDFTraining.GradientBoostedTreesLearner(
        'col_label',
        'CLASSIFICATION',
      );
    const data = {
      'col_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      'col_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
      'col_label': ['x', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'x'],
    };
    const model = learner.train(data);
    const predictions = model.predict(data);
    expect(predictions).toHaveSize(10);
    model.unload();
  });

  it('trains a CART model on an in-memory dataset', () => {
    expect(YDFTraining).not.toBe(null);
    const learner: YDFTrainingwrapper.CartLearner = new YDFTraining.CartLearner(
      'col_label',
      'CLASSIFICATION',
    );
    const data = {
      'col_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      'col_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
      'col_label': ['x', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'x'],
    };
    const model = learner.train(data);
    const predictions = model.predict(data);
    expect(predictions).toHaveSize(10);
    model.unload();
  });

  it('in-memory dataset and csv dataset models match', async () => {
    expect(YDFTraining).not.toBe(null);
    const learner: YDFTrainingwrapper.GradientBoostedTreesLearner =
      new YDFTraining.GradientBoostedTreesLearner('Rings', 'REGRESSION');
    const abaloneCSV = await fetch(abaloneUrl)
      .then((r) => r.blob())
      .then((b) => b.text());
    // Manually convert to dict of columns
    const abaloneDict = csvToDictionary(abaloneCSV);
    const modelFromCsv = learner.train(abaloneCSV);
    const predictionsFromCsv = modelFromCsv.predict(abaloneCSV);
    const modelFromDict = learner.train(abaloneDict);
    const predictionsFromDict = modelFromDict.predict(abaloneDict);
    expect(predictionsFromCsv).toEqual(predictionsFromDict);
    modelFromCsv.unload();
    modelFromDict.unload();
  });

  it('describes a model', () => {
    expect(YDFTraining).not.toBe(null);
    const learner: YDFTrainingwrapper.CartLearner = new YDFTraining.CartLearner(
      'col_label',
      'CLASSIFICATION',
    );
    const data = {
      'col_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      'col_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
      'col_label': ['x', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'x'],
    };
    const model = learner.train(data);
    const expectedDescription = `Type: "RANDOM_FOREST"
Task: CLASSIFICATION
Label: "col_label"

Input Features (2):
	col_1
	col_2

No weights

Variable Importance: INV_MEAN_MIN_DEPTH:

Variable Importance: NUM_AS_ROOT:

Variable Importance: NUM_NODES:

Variable Importance: SUM_SCORE:


Cannot compute model self evaluation:This model does not support evaluation reports.

Winner takes all: false
Out-of-bag evaluation disabled.
Number of trees: 1
Total number of nodes: 1

Number of nodes by tree:
Count: 1 Average: 1 StdDev: 0
Min: 1 Max: 1 Ignored: 0
----------------------------------------------
[ 1, 1] 1 100.00% 100.00% ##########

Depth by leafs:
Count: 1 Average: 0 StdDev: 0
Min: 0 Max: 0 Ignored: 0
----------------------------------------------
[ 0, 0] 1 100.00% 100.00% ##########

Number of training obs by leaf:
Count: 1 Average: 10 StdDev: 0
Min: 10 Max: 10 Ignored: 0
----------------------------------------------
[ 10, 10] 1 100.00% 100.00% ##########

Attribute in nodes:

Attribute in nodes with depth <= 0:

Attribute in nodes with depth <= 1:

Attribute in nodes with depth <= 2:

Attribute in nodes with depth <= 3:

Attribute in nodes with depth <= 5:

Condition type in nodes:
Condition type in nodes with depth <= 0:
Condition type in nodes with depth <= 1:
Condition type in nodes with depth <= 2:
Condition type in nodes with depth <= 3:
Condition type in nodes with depth <= 5:
Node format: NOT_SET
`;
    expect(model.describe()).toBe(expectedDescription);
    model.unload();
  });

  it('saves a model to ZIP', async () => {
    expect(YDFTraining).not.toBe(null);
    const learner: YDFTrainingwrapper.CartLearner = new YDFTraining.CartLearner(
      'col_label',
      'CLASSIFICATION',
    );
    const data = {
      'col_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      'col_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
      'col_label': ['x', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'x'],
    };
    const model = learner.train(data);
    const predictions = model.predict(data);
    const blob = await model.save();
    const loadedModel = await YDFTraining.loadModelFromZipBlob(blob);
    const loadedPredictions = loadedModel.predict(data);
    expect(loadedPredictions).toEqual(predictions);

    model.unload();
    loadedModel.unload();
  });

  afterAll(async () => {});
});
