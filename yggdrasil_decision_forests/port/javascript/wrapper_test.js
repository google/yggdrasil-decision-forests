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
  const modelUrl = '/base/third_party/yggdrasil_decision_forests/port/javascript/example/model.zip'
  const modelOptions = { createdTFDFSignature: true };

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

    // Load model
    await ydf
        .loadModelFromUrl(modelUrl, modelOptions)
        .then((loadedModel) => {
          model = loadedModel;
          console.log('Model loaded');
        });
  });

  it('loadModelFromUrl', () => {
    expect(model).not.toBeNull();
  });

  it('loadModelFromZipBlob', async () => {
    let blobModel = null;

    const modelBlob = await fetch(modelUrl).then(r => r.blob());
    await ydf.loadModelFromZipBlob(modelBlob, modelOptions).then((loadedModel) => {
      model = loadedModel;
    });

    expect(model).not.toBeNull();
  });

  it('loadModelFromZipBlobWithoutarrayBuffer()', async () => {
    let blobModel = null;
    const modelBlob = await fetch(modelUrl).then(r => r.blob());

    // Force model to use FileReader conversion.
    const arrayBufferFunc = Blob.prototype.arrayBuffer;
    Blob.prototype.arrayBuffer = undefined;

    await ydf.loadModelFromZipBlob(modelBlob, modelOptions).then((loadedModel) => {
      model = loadedModel;
    });

    Blob.prototype.arrayBuffer = arrayBufferFunc;

    expect(model).not.toBeNull();
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
});
