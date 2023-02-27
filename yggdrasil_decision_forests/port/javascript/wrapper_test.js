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

describe('YDF Inference', () => {
  let ydf = null;

  let model = null;
  const modelUrl =
      '/base/third_party/yggdrasil_decision_forests/port/javascript/example/model.zip';

  let model2 = null;
  const modelUrl2 =
      '/base/third_party/yggdrasil_decision_forests/port/javascript/test_data/model_2.zip';

  const modelUrl3 =
      '/base/third_party/yggdrasil_decision_forests/port/javascript/test_data/model_3.zip';

  let modelSmallSST = null;
  const modelSmallSSTUrl =
      '/base/third_party/yggdrasil_decision_forests/port/javascript/test_data/model_small_sst.zip';

  const modelOptions = {createdTFDFSignature: true};

  /**
   * Ensures that array "actual" is close to array "expected".
   */
  function expectToBeCloseToArray(actual, expected) {
    expect(actual.length).toBe(expected.length);
    actual.forEach(
        (x, i) => expect(x).withContext(`[${i}]`).toBeCloseTo(expected[i]));
  }

  /**
   * Ensures that matrix "actual" is close to array "expected".
   */
  function expectToBeCloseToMatrix(actual, expected) {
    expect(actual.length).toBe(expected.length);
    actual.forEach((x, i) => expectToBeCloseToArray(x, expected[i]));
  }

  beforeAll(async () => {
    // Load library
    await YggdrasilDecisionForests({
      locateFile: function(filename, dir) {
        return dir + filename;
      }
    }).then(function(m) {
      ydf = m;
      console.log('The library is loaded');
    });

    // Load models
    await ydf.loadModelFromUrl(modelUrl, modelOptions).then((loadedModel) => {
      model = loadedModel;
      console.log('Model 1 loaded');
    });

    await ydf.loadModelFromUrl(modelUrl2, modelOptions).then((loadedModel) => {
      model2 = loadedModel;
      console.log('Model 2 loaded');
    });

    await ydf.loadModelFromUrl(modelSmallSSTUrl, modelOptions)
        .then((loadedModel) => {
          modelSmallSST = loadedModel;
          console.log('Model Small SST loaded');
        });
  });

  it('loadModelFromUrl', () => {
    expect(model).not.toBeNull();
    expect(model2).not.toBeNull();
    expect(modelSmallSST).not.toBeNull();
  });

  it('loadModelFromZipBlob', async () => {
    const modelBlob = await fetch(modelUrl).then(r => r.blob());
    let my_model = null;
    await ydf.loadModelFromZipBlob(modelBlob, modelOptions)
        .then((loadedModel) => {
          my_model = loadedModel;
        });

    expect(my_model).not.toBeNull();
    my_model.unload();
  });

  it('loadModelFromZipBlobWithPrefixEmpty', async () => {
    const modelBlob = await fetch(modelUrl).then(r => r.blob());
    let my_model = null;
    await ydf
        .loadModelFromZipBlob(
            modelBlob, {createdTFDFSignature: true, file_prefix: ''})
        .then((loadedModel) => {
          my_model = loadedModel;
        });

    expect(my_model).not.toBeNull();
    my_model.unload();
  });

  it('loadModelFromZipBlobWithPrefixManual', async () => {
    const modelBlob = await fetch(modelUrl3).then(r => r.blob());
    let model_1 = null;
    let model_2 = null;
    let model_3 = null;

    console.log('Load model3 with prefix \'\'');
    await ydf
        .loadModelFromZipBlob(
            modelBlob, {createdTFDFSignature: true, file_prefix: ''})
        .then((loadedModel) => {
          model_1 = loadedModel;
        });

    console.log('Load model3 with prefix \'p1_\'');
    await ydf
        .loadModelFromZipBlob(
            modelBlob, {createdTFDFSignature: true, file_prefix: 'p1_'})
        .then((loadedModel) => {
          model_2 = loadedModel;
        });

    console.log('Load model3 with prefix \'p2_\'');
    await ydf
        .loadModelFromZipBlob(
            modelBlob, {createdTFDFSignature: true, file_prefix: 'p2_'})
        .then((loadedModel) => {
          model_3 = loadedModel;
        });

    expect(model_1).not.toBeNull();
    expect(model_2).not.toBeNull();
    expect(model_3).not.toBeNull();
    model_1.unload();
    model_2.unload();
    model_3.unload();
  });

  it('loadModelFromZipBlobWithoutarrayBuffer()', async () => {
    const modelBlob = await fetch(modelUrl).then(r => r.blob());

    // Force model to use FileReader conversion.
    const arrayBufferFunc = Blob.prototype.arrayBuffer;
    Blob.prototype.arrayBuffer = undefined;

    let my_model = null;
    await ydf.loadModelFromZipBlob(modelBlob, modelOptions)
        .then((loadedModel) => {
          my_model = loadedModel;
        });

    Blob.prototype.arrayBuffer = arrayBufferFunc;

    expect(my_model).not.toBeNull();
    my_model.unload();
  });

  it('predict', async () => {
    let predictions = model.predict({
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

    // Ground truth predictions generated with:
    //
    // ::shell
    // unzip
    // third_party/yggdrasil_decision_forests/port/javascript/example/model.zip
    // -d /tmp/model_extracted
    //
    // bazel build -c opt //third_party/yggdrasil_decision_forests/cli:predict
    //
    // bazel-bin/third_party/yggdrasil_decision_forests/cli/predict
    // --model=/tmp/model_extracted
    // --dataset=csv:third_party/yggdrasil_decision_forests/test_data/dataset/adult_test.csv
    // --output=csv:/tmp/predictions.csv
    //
    // head /tmp/predictions.csv
    expect(predictions).toEqual([
      0.13323983550071716,
      0.47678571939468384,
      0.818461537361145,
      0.4974619150161743,
    ]);
  });

  it('predict_catset', async () => {
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

    // The model is:
    //
    // Tree #0:
    //     pred:0
    //
    // Tree #1:
    //     "f3" is in [BITMAP] {Z} [s:0.15 n:8 np:3 miss:0] ; pred:0
    //         ├─(pos)─ pred:0.2
    //         └─(neg)─ pred:-0.12
    //
    // Tree #2:
    //     pred:1.601e-05
    //
    // Tree #3:
    //     pred:1.4409e-05
    //
    // Tree #4:
    //     "f2" is in [BITMAP] {RED} [s:9.94956e-05 n:8 np:4 miss:0] ;
    //     pred:1.29579e-05
    //         ├─(pos)─ pred:-0.0040041
    //         └─(neg)─ "f3" is in [BITMAP] {Z} [s:0.0645778 n:4 np:1 miss:0] ;
    //         pred:0.00402361
    //                  ├─(pos)─ pred:0.181871
    //                  └─(neg)─ pred:-0.0548811

    expectToBeCloseToArray(predictions, [
      0.4690462, 0.4563984, 0.4563984, 0.54885024, 0.4563984, 0.5943315,
      0.4690462, 0.54885024
    ]);
  });


  it('predictTensorFlowDecisionForestSignature', async () => {
    // The signature has been generated using the
    // "tfdf.tensorflow.ops.inference.api._InferenceArgsBuilder()" class in the
    // test data model contained in TF-DF
    // (test_data/model/saved_model_adult_gbt).
    //
    // ::python
    // test_df = pd.read_csv(gfile.Open("adult_test.csv","r"))
    // features = test_df[:4].fillna("").to_dict(orient="list")
    // features = {k:tf.constant(v) for k,v in features.items()}
    //
    // arg_builder = tfdf.tensorflow.ops.inference.api._InferenceArgsBuilder()
    // arg_builder.build_from_model_path("/my_model/assets")
    // arg_builder.init_op()
    // args = arg_builder.build_inference_op_args(features)
    // del args["dense_output_dim"]
    // args = {k:v.numpy().tolist() for k,v in args.items()}
    // args

    let predictions = model.predictTFDFSignature({
      numericalFeatures: [
        [39.0, 2174.0, 0.0, 13.0, 77516.0, 40.0],
        [40.0, 0.0, 0.0, 11.0, 121772.0, 40.0],
        [40.0, 0.0, 0.0, 16.0, 193524.0, 60.0],
        [35.0, 0.0, 0.0, 5.0, 76845.0, 40.0]
      ],
      booleanFeatures: [],
      categoricalIntFeatures: [
        [3, 2, 1, 4, 1, 2, 1, 4],
        [5, 1, -1, 3, 3, 1, 1, 1],
        [13, 1, 1, 1, 1, 1, 1, 1],
        [11, 1, 1, 10, 2, 1, 1, 6],
      ],
      categoricalSetIntFeaturesValues: [],
      categoricalSetIntFeaturesRowSplitsDim1: [0],
      categoricalSetIntFeaturesRowSplitsDim2: [0],
      denseOutputDim: 1,
    });
    console.log('Predictions:', predictions);

    // Same ground truth as "predict".
    expect(predictions.densePredictions).toEqual([
      [0.8667601346969604, 0.13323983550071716],
      [0.5232142806053162, 0.47678571939468384],
      [0.18153846263885498, 0.818461537361145],
      [0.5025380849838257, 0.4974619150161743]
    ]);

    expect(predictions.denseColRepresentation).toEqual(['1', '2']);
  });

  it('predictTensorFlowDecisionForestSignatureCatSet', async () => {
    let predictions = model2.predictTFDFSignature({
      booleanFeatures: [],
      categoricalIntFeatures: [],
      categoricalSetIntFeaturesRowSplitsDim1:
          [0, 2, 3, 4, 5, 5, 5, 8, 11, 12, 13, 13, 15, 16, 17, 19, 20],
      categoricalSetIntFeaturesRowSplitsDim2: [0, 2, 4, 6, 8, 10, 12, 14, 16],
      categoricalSetIntFeaturesValues:
          [1, 2, 3, 3, 1, 1, 2, 3, 3, 1, 2, 2, 3, 2, 1, 1, 1, 2, 1, 2],
      numericalFeatures:
          [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
      denseOutputDim: 1,
    });
    console.log('Predictions:', predictions);

    // Same ground truth as "predict".
    expectToBeCloseToMatrix(predictions.densePredictions, [
      [0.53095376, 0.4690462], [0.54360163, 0.4563984], [0.54360163, 0.4563984],
      [0.45114976, 0.54885024], [0.54360163, 0.4563984], [0.4056685, 0.5943315],
      [0.53095376, 0.4690462], [0.45114976, 0.54885024]
    ]);

    expect(predictions.denseColRepresentation).toEqual(['1', '2']);
  });

  it('predictTensorFlowDecisionForestSignatureCatSetSmallSST', async () => {
    // Run on 20 examples.

    // Example generated with
    // tfdf.tensorflow.ops.inference.api._InferenceArgsBuilder().
    let predictions = modelSmallSST.predictTFDFSignature({
      booleanFeatures: [],
      categoricalIntFeatures: [],
      categoricalSetIntFeaturesRowSplitsDim1: [
        0,   12,  42,  58,  85,  104, 132, 137, 151, 158, 174,
        194, 209, 221, 241, 258, 281, 308, 315, 334, 362
      ],
      categoricalSetIntFeaturesRowSplitsDim2: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20
      ],
      categoricalSetIntFeaturesValues: [
        3,  0,  0,  0,  7,  0,  0,  0,  26, 1,  0,  6,  0,  3,  0,  0,  0,  36,
        1,  0,  0,  0,  2,  74, 0,  76, 0,  1,  0,  0,  13, 0,  8,  74, 0,  76,
        14, 27, 0,  0,  0,  6,  55, 0,  0,  10, 55, 0,  0,  7,  0,  1,  0,  0,
        5,  1,  58, 6,  1,  0,  0,  26, 0,  0,  0,  3,  0,  5,  0,  2,  0,  50,
        63, 2,  22, 0,  0,  0,  2,  0,  2,  26, 1,  0,  6,  36, 1,  0,  0,  2,
        12, 8,  0,  10, 41, 34, 1,  0,  9,  3,  0,  0,  19, 6,  0,  0,  11, 0,
        0,  4,  0,  2,  20, 0,  0,  9,  0,  0,  26, 1,  0,  0,  0,  0,  4,  1,
        0,  22, 0,  0,  0,  6,  0,  0,  5,  0,  6,  29, 5,  1,  27, 0,  0,  8,
        94, 7,  0,  0,  20, 0,  6,  22, 12, 0,  42, 83, 0,  6,  1,  19, 8,  0,
        0,  0,  4,  0,  0,  0,  0,  14, 1,  0,  0,  6,  12, 62, 25, 0,  11, 0,
        2,  12, 33, 61, 0,  5,  99, 47, 12, 8,  59, 0,  0,  6,  22, 12, 8,  50,
        0,  4,  50, 0,  4,  12, 0,  11, 3,  0,  6,  0,  0,  33, 27, 31, 87, 0,
        7,  85, 12, 0,  6,  11, 1,  0,  2,  1,  19, 0,  24, 17, 0,  0,  0,  1,
        80, 0,  5,  0,  0,  0,  6,  3,  0,  5,  0,  4,  0,  10, 0,  0,  0,  11,
        17, 0,  0,  4,  0,  6,  0,  11, 0,  53, 11, 0,  0,  0,  0,  7,  0,  5,
        17, 0,  13, 0,  27, 31, 0,  11, 3,  0,  6,  64, 9,  0,  0,  0,  0,  9,
        3,  0,  5,  65, 1,  43, 0,  0,  2,  15, 0,  5,  0,  0,  2,  4,  0,  0,
        0,  6,  27, 0,  0,  31, 0,  0,  6,  0,  26, 20, 0,  16, 7,  0,  3,  0,
        37, 39, 0,  10, 98, 0,  0,  1,  0,  6,  12, 8,  0,  7,  0,  1,  0,  10,
        0,  0,  3,  0,  2,  22, 0,  8,  0,  10, 0,  0,  4,  0,  0,  0,  3,  0,
        0,  6
      ],
      numericalFeatures: [],
      denseOutputDim: 1,
    });
    console.log('Predictions:', predictions);

    // Ground truth predictions generated with model.predict (python api).
    expectToBeCloseToMatrix(predictions.densePredictions, [
      [0.0999999, 0.9000001],
      [0.0999999, 0.9000001],
      [0.19999993, 0.8000001],
      [0.6, 0.4],
      [0.29999995, 0.70000005],
      [0.29999995, 0.70000005],
      [0., 1.],
      [0.6, 0.4],
      [0.9, 0.1],
      [0.19999993, 0.8000001],
      [0.6, 0.4],
      [0.6, 0.4],
      [0.39999998, 0.6],
      [0.29999995, 0.70000005],
      [0., 1.],
      [0.29999995, 0.70000005],
      [0.0999999, 0.9000001],
      [0.29999995, 0.70000005],
      [0.29999995, 0.70000005],
      [0.0999999, 0.9000001]
    ]);

    expect(predictions.denseColRepresentation).toEqual(['1', '2']);
  });

  afterAll(async () => {
    model.unload();
    model2.unload();
    modelSmallSST.unload();
  });
});
