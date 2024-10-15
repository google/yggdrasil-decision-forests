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

const fs = require('node:fs');

describe('YDF Inference', () => {
  let ydf = null;
  let model1 = null;
  let model2 = null;

  beforeAll(async () => {
    ydf = await require('ydf-inference')();

    model1 =
        await ydf.loadModelFromZipBlob(fs.readFileSync('./test/model_1.zip'));
    model2 =
        await ydf.loadModelFromZipBlob(fs.readFileSync('./test/model_2.zip'));
  });

  it('loadModelFromZipBlob', () => {
    expect(model1).not.toBeNull();
    expect(model2).not.toBeNull();
  });

  it('predict model1', async () => {
    let predictions = model1.predict({
      'age': [39, 40, 40, 35],
      'workclass': ['State-gov', 'Private', 'Private', 'Federal-gov'],
      'fnlwgt': [77516, 121772, 193524, 76845],
      'education': ['Bachelors', 'Assoc-voc', 'Doctorate', '9th'],
      'education_num': [13, 11, 16, 5],
      'marital_status': [
        'Never-married', 'Married-civ-spouse', 'Married-civ-spouse',
        'Married-civ-spouse'
      ],
      'occupation':
          ['Adm-clerical', 'Craft-repair', 'Prof-specialty', 'Farming-fishing'],
      'relationship': ['Not-in-family', 'Husband', 'Husband', 'Husband'],
      'race': ['White', 'Asian-Pac-Islander', 'White', 'Black'],
      'sex': ['Male', 'Male', 'Male', 'Male'],
      'capital_gain': [2174, 0, 0, 0],
      'capital_loss': [0, 0, 0, 0],
      'hours_per_week': [40, 40, 60, 40],
      'native_country':
          ['United-States', null, 'United-States', 'United-States']
    });
    console.log('Predictions:', predictions);

    expect(predictions).toEqual([
      0.13323983550071716,
      0.47678571939468384,
      0.818461537361145,
      0.4974619150161743,
    ]);
  });

  it('predict model2', async () => {
    let predictions = model2.predict({
      'f1': [0, 0, 0, 0, 0, 0, 0, 0],
      'f2': [
        ['RED', 'BLUE'], ['GREEN'], [], ['RED', 'BLUE', 'GREEN'], ['BLUE'], [],
        ['RED'], ['BLUE', 'RED']
      ],
      'f3':
          [['X'], ['Y'], [], ['X', 'Y', 'Z'], ['X'], ['Z', 'Y'], ['Y'], ['Z']],
    });
    console.log('Predictions:', predictions);
    expect(predictions).toEqual([
      0.4690462052822113,
      0.4563983976840973,
      0.4563983976840973,
      0.5488502383232117,
      0.4563983976840973,
      0.5943315029144287,
      0.4690462052822113,
      0.5488502383232117,
    ]);
  });

  afterAll(async () => {
    model1.unload();
    model2.unload();
  });
});
