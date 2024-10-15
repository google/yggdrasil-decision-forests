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

describe('YDF Training', () => {
  beforeAll(async () => {
    ydf = await require('ydf-training')({
      "print": console.log,
      "printErr": console.log,
    });
  });

  it('train GBT regression', () => {
    abalone_csv = fs.readFileSync('./test/abalone.csv', 'utf-8');
    const learner = new ydf.GradientBoostedTreesLearner("Rings", "REGRESSION");
    const model = learner.train(abalone_csv);
    const description = model.describe();
    const expectedStart = 'Type: "GRADIENT_BOOSTED_TREES"\nTask: REGRESSION\nLabel: "Rings"\n\nInput Features (8):\n\tType\n\tLongestShell\n\tDiameter\n\tHeight\n\tWholeWeight\n\tShuckedWeight\n\tVisceraWeight\n\tShellWeight\n\nNo weights';
    expect(description.startsWith(expectedStart)).toBe(true);
    const regex = /Number of trees: (\d+)/;
    const match = description.match(regex);
    expect(match).not.toBeNull(); // Check if the pattern is found
    if (match) {
      const numTrees = parseInt(match[1], 10);
      expect(numTrees).toBeGreaterThan(20); 
    }
  });

  it('train RF regression', () => {
    abalone_csv = fs.readFileSync('./test/abalone.csv', 'utf-8');
    const learner = new ydf.RandomForestLearner("Rings", "REGRESSION");
    const model = learner.train(abalone_csv);
    const description = model.describe();
    const expectedStart = 'Type: "RANDOM_FOREST"\nTask: REGRESSION\nLabel: "Rings"\n\nInput Features (8):\n\tType\n\tLongestShell\n\tDiameter\n\tHeight\n\tWholeWeight\n\tShuckedWeight\n\tVisceraWeight\n\tShellWeight\n\nNo weights';
    expect(description.startsWith(expectedStart)).toBe(true);
    expect(description).toContain("Number of trees: 300");
  });

  afterAll(async () => {
  });
});
