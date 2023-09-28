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
 * @fileoverview JavaScript native inference for Yggdrasil Decision Forests
 * models.
 *
 * See "README.md" and "/example" for two usage examples.
 */

/**
 * Semantic of the input features. Those fields are equivalent to the
 * "ColumnType" proto enum defined in
 * "yggdrasil_decision_forests/dataset/data_spec.proto".
 * @enum {string}
 */
const ColumnType = {
  NUMERICAL: 'NUMERICAL',
  CATEGORICAL: 'CATEGORICAL',
  CATEGORICAL_SET: 'CATEGORICAL_SET',
};

/**
 * A list of input examples.
 *
 * Examples represents a list of examples. It is an object indexed by feature
 * names. For each feature, the object contains an array of values indexed by
 * example. In those arrays, the value null represent missing values.
 *
 * Example:
 *
 *  A list of two examples. The first example features are: f1=1 and
 *  f2=red. The second examples features are f1=2 and f2 is missing.
 *  const examples = {
 *     "f1": [1, 2],
 *     "f2": ["red", null],
 *    };
 *
 * @typedef {!Object<string, (!Array<number>|!Array<string>)>}
 */
let Examples;

/**
 * Build the index from index in the TF-DF signature to serving (a.k.a internal)
 * feature index.
 *
 * @param {!Array<!InputFeature>} protoInputFeatures Input features of the
 *     model.
 * @param {!Array<!InputFeature>} inputFeatures Input features of the engine.
 * @param {!Array<string>} types Feature types to index.
 * @return {!Array<number>} Indices of the features for the inference engine.
 */
function indexTFDFFeatures(protoInputFeatures, inputFeatures, types) {
  // List the pair of (specIdx,internalIdx) for all the input features.

  let indices = [];
  for (const protoFeatureDef of protoInputFeatures) {
    if (!types.includes(protoFeatureDef.type)) {
      continue;
    }

    // Look for the index of the feature for the inference engine.
    let internalIdx = -1;
    for (const engineFeatureDef of inputFeatures) {
      if (protoFeatureDef.specIdx == engineFeatureDef.specIdx) {
        internalIdx = engineFeatureDef.internalIdx;
        break;
      }
    }
    // internalIdx == -1 indicates that the feature is not used by the engine.

    indices.push({
      'specIdx': protoFeatureDef.specIdx,
      'internalIdx': internalIdx,
    });
  }

  // Sort the features in specIdx. This is the order of the TF-DF signature.
  indices.sort((a, b) => (a.specIdx < b.specIdx) ? -1 : 1);

  // Extract and return the internal index.
  return Array.from({length: indices.length})
      .map((unused, index) => indices[index].internalIdx);
}


/**
 * Converts a std::vector<T> (C++) into a JS array.
 * @param {!CCVector} src CC Vector.
 * @return {!Array} JS Vector.
 */
function ccVectorToJSVector(src) {
  return Array.from({length: src.size()})
      .map((unused, index) => src.get(index));
}

/**
 * Converts a JS array into a std::vector<std::string> (C++).
 * @param {!Array} src JS Vector.
 * @return {!CCVector} CC Vector.
 */
function jsStrVectorToCCStrVector(src) {
  const vector = Module.CreateVectorString(src.length);
  for (const value of src) {
    vector.push_back(value);
  }
  return vector;
}

/**
 * Converts a JS array into a std::vector<int> (C++).
 * @param {!Array} src JS Vector.
 * @return {!CCVector} CC Vector.
 */
function jsStrVectorToCCIntVector(src) {
  const vector = Module.CreateVectorInt(src.length);
  for (const value of src) {
    vector.push_back(value);
  }
  return vector;
}

/**
 * Converts a std::vector<std::vector<T>> (C++) into a JS array or array.
 * @param {!CCVectorVector} src CC Matrix.
 * @return {!Array<!Array>} JS Matrix.
 */
function ccMatrixToJSMatrix(src) {
  return Array.from({length: src.size()})
      .map((unused, index) => ccVectorToJSVector(src.get(index)));
}

/**
 * Converts a Blob into a promise resolving to an arrayBuffer.
 * @param {!Blob} blob Input file.
 * @return {!Promise<ArrayBuffer>} promise resolving to arrayBuffer contents.
 */
function blobToArrayBuffer(blob) {
  if (blob.arrayBuffer) {
    return blob.arrayBuffer();
  } else {
    return new Promise(resolve => {
      const fileReader = new FileReader();

      fileReader.readAsArrayBuffer(blob);
      fileReader.onload = function(event) {
        resolve(event.target.result);
      }
    });
  }
}

/**
 * A machine learning model.
 */
class Model {
  /**
   * Creates a model.
   * @param {!InternalModel} internalModel The internal cc/wasm model.
   * @param {boolean} createdTFDFSignature If true, create the TF-DF signature
   *     of the model.
   */
  constructor(internalModel, createdTFDFSignature) {
    /** @private {?InternalModel} */
    this.internalModel = internalModel;
    this.createdTFDFSignature = createdTFDFSignature;

    const rawInputFeatures = this.internalModel.getInputFeatures();
    /**
     * List of input features of the model.
     * @const @private @type {!Array<!InputFeature>}
     */
    this.inputFeatures =
        Array.from({length: rawInputFeatures.size()})
            .map((unused, index) => rawInputFeatures.get(index));

    /**
     * Index of the numerical input features for the TF-DF signature.
     * @private @type {?Array<number>}
     */
    this.numericalFeaturesIndex = null;

    /**
     * Index of the boolean input features for the TF-DF signature.
     * @private @type {?Array<number>}
     */
    this.booleanFeaturesIndex = null;

    /**
     * Index of the categorical input features for the TF-DF signature.
     * @private @type {?Array<number>}
     */
    this.categoricalIntFeaturesIndex = null;

    /**
     * Index of the categorical-set input features for the TF-DF signature.
     * @private @type {?Array<number>}
     */
    this.categoricalSetIntFeaturesIndex = null;

    if (this.createdTFDFSignature) {
      this.createdTFDFSignature_();
    }
  }

  /**
   * Create the TF-DF signature of the model.
   */
  createdTFDFSignature_() {
    // Index the input features for the TensorFlow Decision Forests signature.
    const rawProtoInputFeatures = this.internalModel.getProtoInputFeatures();
    const protoInputFeatures = ccVectorToJSVector(rawProtoInputFeatures);
    rawProtoInputFeatures.delete();

    this.numericalFeaturesIndex = indexTFDFFeatures(
        protoInputFeatures, this.inputFeatures,
        ['NUMERICAL', 'DISCRETIZED_NUMERICAL']);
    this.booleanFeaturesIndex =
        indexTFDFFeatures(protoInputFeatures, this.inputFeatures, ['BOOLEAN']);
    this.categoricalIntFeaturesIndex = indexTFDFFeatures(
        protoInputFeatures, this.inputFeatures, ['CATEGORICAL']);
    this.categoricalSetIntFeaturesIndex = indexTFDFFeatures(
        protoInputFeatures, this.inputFeatures, ['CATEGORICAL_SET']);
  }


  /**
   * Lists the input features of the model.
   * @return {!Array<!InputFeature>} List of input features of the model.
   */
  getInputFeatures() {
    return this.inputFeatures;
  }

  /**
   * Applies the model on a list of examples and returns the predictions.
   *
   * Usage example:
   *
   *  // A list of two examples. The first example features are: f1=1 and
   *  // f2=red. The second examples features are f1=2 and f2 is missing.
   *  const examples = {
   *     "f1": [1 ,2],
   *     "f2": ["red" ,null],
   *    };
   *
   *  const predictions = model.predict(examples);
   *  // If the model's output dimension is 1 (e.g. the model is a binary
   *  // classifier configured to return the probability of the "positive"
   *  // class), "predictions[0]" and "predictions[1]" are respectively the
   *  // probability predictions of the first and second examples.
   *
   * @param {!Examples} examples A list of examples represented by a single
   *     object containing one attribute for each of the input features of the
   *     model. Each field is an array containing a value for each of the
   *     the value null represent missing values.
   * @return {!Array<number>} The predictions of the model.
   */
  predict(examples) {
    if (typeof examples !== 'object') {
      throw new Error('argument should be an array or an object');
    }

    // Detect the number of examples and ensure that all the fields (i.e.
    // features) are arrays with the same number of items.
    let numExamples = undefined;
    for (const values of Object.values(examples)) {
      if (!Array.isArray(values)) {
        throw new Error('features should be arrays');
      }
      if (numExamples === undefined) {
        numExamples = values.length;
      } else if (numExamples !== values.length) {
        throw new Error('features have a different number of values');
      }
    }
    if (numExamples === undefined) {
      // The example does not contain any features.
      throw new Error('not features');
    }

    // Fill the examples
    this.internalModel.newBatchOfExamples(numExamples);
    for (const featureDef of this.inputFeatures) {
      const values = examples[featureDef.name];
      if (featureDef.type === ColumnType.NUMERICAL) {
        for (const [exampleIdx, value] of values.entries()) {
          if (value === null) continue;
          this.internalModel.setNumerical(
              exampleIdx, featureDef.internalIdx, value);
        }
      } else if (featureDef.type === ColumnType.CATEGORICAL) {
        for (const [exampleIdx, value] of values.entries()) {
          if (value === null) continue;
          if (typeof value === 'string') {
            this.internalModel.setCategoricalString(
                exampleIdx, featureDef.internalIdx, value);
          } else {
            this.internalModel.setCategoricalInt(
                exampleIdx, featureDef.internalIdx, value);
          }
        }
      } else if (featureDef.type === ColumnType.CATEGORICAL_SET) {
        for (const [exampleIdx, value] of values.entries()) {
          if (value === null) continue;
          const cc_values = jsStrVectorToCCStrVector(value);
          this.internalModel.setCategoricalSetString(
              exampleIdx, featureDef.internalIdx, cc_values);
          cc_values.delete();
        }
      } else {
        throw new Error(`Non supported feature type ${featureDef}`);
      }
    }

    // Extract predictions
    const internalPredictions = this.internalModel.predict();
    const jsPredictions = ccVectorToJSVector(internalPredictions);
    internalPredictions.delete();
    return jsPredictions;
  }

  /**
   * Applies the model on a list of examples given in the format of the
   * TensorFlow Decision Forests inference ops called "SimpleMLInferenceOp*" and
   * build by the "_InferenceArgsBuilder" utility. Require for the model to be
   * loaded with "createdTFDFSignature=true".
   *
   * See tensorflow_decision_forests/tensorflow/ops/inference/op.cc for the
   * definition of the format. For scalar features, the format is simply V_i,j
   * where i is the example index and j means the j-th feature of a specific
   * type (e.g. numerical for the "numericalFeatures" argument) sorted by
   * dataspec column index.
   *
   * @param {!TFDFInput} inputs Input features.
   * @return {!TFDFOutputPrediction} Predictions of the model.
   */
  predictTFDFSignature(inputs) {
    if (!this.createdTFDFSignature) {
      throw new Error(
          'Model not loaded with options.createdTFDFSignature=true');
    }

    // Detect the number of examples.
    //
    // For each type of given feature, extract the number of examples. Ensure
    // that the feature shapes are consistent.
    let numExamples = 0;
    if (inputs.numericalFeatures.length != 0) {
      if (numExamples != 0 && numExamples != inputs.numericalFeatures.length) {
        throw new Error('features have a different number of values');
      }
      if (this.numericalFeaturesIndex.length !=
          inputs.numericalFeatures[0].length) {
        throw new Error('Unexpected numerical input feature shape');
      }
      numExamples = inputs.numericalFeatures.length;
    }
    if (inputs.booleanFeatures.length != 0) {
      if (numExamples != 0 && numExamples != inputs.booleanFeatures.length) {
        throw new Error('features have a different number of values');
      }
      if (this.booleanFeaturesIndex.length !=
          inputs.booleanFeatures[0].length) {
        throw new Error('Unexpected boolean input feature shape');
      }
      numExamples = inputs.booleanFeatures.length;
    }
    if (inputs.categoricalIntFeatures.length != 0) {
      if (numExamples != 0 &&
          numExamples != inputs.categoricalIntFeatures.length) {
        throw new Error('features have a different number of values');
      }
      if (this.categoricalIntFeaturesIndex.length !=
          inputs.categoricalIntFeatures[0].length) {
        throw new Error('Unexpected categorical int input feature shape');
      }
      numExamples = inputs.categoricalIntFeatures.length;
    }
    if (inputs.categoricalSetIntFeaturesRowSplitsDim2.length > 1) {
      if (this.categoricalSetIntFeaturesIndex.length == null ||
          this.categoricalSetIntFeaturesIndex.length <= 0) {
        throw new Error('Invalid categoricalSetIntFeaturesIndex');
      }
      const detectedNumExamples =
          (inputs.categoricalSetIntFeaturesRowSplitsDim2
               [inputs.categoricalSetIntFeaturesRowSplitsDim2.length - 1] /
           this.categoricalSetIntFeaturesIndex.length);
      if (numExamples != 0 && numExamples != detectedNumExamples) {
        throw new Error('Invalid categorical-set feature shape');
      }
      numExamples = detectedNumExamples;
    }

    // Allocate the examples
    this.internalModel.newBatchOfExamples(numExamples);

    // Set the example values.
    //
    // In the following loops, 'localIdx' is always the index of the feature in
    // the array provided as argument of this function.
    for (let localIdx = 0; localIdx < this.numericalFeaturesIndex.length;
         localIdx++) {
      const internIdx = this.numericalFeaturesIndex[localIdx];
      if (internIdx == -1) {
        continue;
      }
      for (let exampleIdx = 0; exampleIdx < numExamples; exampleIdx++) {
        let value = inputs.numericalFeatures[exampleIdx][localIdx];
        if (isNaN(value)) {
          continue;
        }
        this.internalModel.setNumerical(exampleIdx, internIdx, value);
      }
    }

    for (let localIdx = 0; localIdx < this.booleanFeaturesIndex.length;
         localIdx++) {
      const internIdx = this.booleanFeaturesIndex[localIdx];
      if (internIdx == -1) {
        continue;
      }
      for (let exampleIdx = 0; exampleIdx < numExamples; exampleIdx++) {
        const value = inputs.booleanFeatures[exampleIdx][localIdx];
        if (isNaN(value)) {
          continue;
        }
        this.internalModel.setBoolean(exampleIdx, internIdx, value);
      }
    }

    for (let localIdx = 0; localIdx < this.categoricalIntFeaturesIndex.length;
         localIdx++) {
      const internIdx = this.categoricalIntFeaturesIndex[localIdx];
      if (internIdx == -1) {
        continue;
      }
      for (let exampleIdx = 0; exampleIdx < numExamples; exampleIdx++) {
        const value = inputs.categoricalIntFeatures[exampleIdx][localIdx];
        if (value < 0) {
          continue;
        }
        this.internalModel.setCategoricalInt(exampleIdx, internIdx, value);
      }
    }

    for (let localIdx = 0;
         localIdx < this.categoricalSetIntFeaturesIndex.length; localIdx++) {
      const internIdx = this.categoricalSetIntFeaturesIndex[localIdx];
      if (internIdx == -1) {
        continue;
      }
      for (let exampleIdx = 0; exampleIdx < numExamples; exampleIdx++) {
        const d1Cell =
            exampleIdx * this.categoricalSetIntFeaturesIndex.length + localIdx;
        const beginIdx = inputs.categoricalSetIntFeaturesRowSplitsDim1[d1Cell];
        const endIdx =
            inputs.categoricalSetIntFeaturesRowSplitsDim1[d1Cell + 1];
        const ccValues = Module.CreateVectorInt(endIdx - beginIdx);

        if (endIdx > beginIdx &&
            inputs.categoricalSetIntFeaturesValues[beginIdx] == -1) {
          // This is a missing value.
          continue;
        }

        for (let itemIdx = beginIdx; itemIdx < endIdx; itemIdx++) {
          ccValues.push_back(inputs.categoricalSetIntFeaturesValues[itemIdx]);
        }
        this.internalModel.setCategoricalSetInt(
            exampleIdx, internIdx, ccValues);
        ccValues.delete();
      }
    }


    // Generate predictions.
    const rawPredictions =
        this.internalModel.predictTFDFSignature(inputs.denseOutputDim);

    // Note: It is unclear why, but "rawPredictions" and its components (e.g.
    // rawPredictions.denseColRepresentation) should not be deleted.

    // Convert predictions to js format.
    const denseColRepresentation =
        ccVectorToJSVector(rawPredictions.denseColRepresentation);
    const densePredictions =
        ccMatrixToJSMatrix(rawPredictions.densePredictions);

    return {
      densePredictions: densePredictions,
      denseColRepresentation: denseColRepresentation,
    };
  }

  /**
   * Unloads the model from memory.
   *
   * Usage example:
   *
   *    model.unload();
   *    model = null;
   *
   * Models (e.g. loaded with "loadModelFromUrl") should be released from
   * memory manually by calling "model.unload()". Not "unloading" the model
   * will result in a memory leak.
   *
   * TODO: Unload the model automatically using "finalizers" once available
   * in JS.
   *
   * See
   * https://emscripten.org/docs/porting/connecting_cpp_and_javascript/embind.html?highlight=pointer%20delete#memory-management
   * for details.
   */
  unload() {
    if (this.internalModel !== null) {
      this.internalModel.delete();
      this.internalModel = null;
    }
  }
}

/**
 * Loads a model from a blob containing a zipped Yggdrasil model.
 *
 * @param {!Object} serializedModel Model zip blob.
 * @param {!LoadModelOptions=} options Loading model options.
 * @return {!Promise<!Model>} The loaded model.
 */
Module['loadModelFromZipBlob'] =
    async function loadModelFromZipBlob(serializedModel, options) {
  // Read options.

  // Note: Emscripten has issues with ES6 spead operator.
  // https://github.com/emscripten-core/emscripten/pull/18461
  options = (options !== undefined) ?
      /** @type {!LoadModelOptions} */ (Object.assign({}, options)) :
      /** @type {!LoadModelOptions} */ ({});

  options.createdTFDFSignature = (options.createdTFDFSignature === undefined) ?
      false :
      options.createdTFDFSignature;
  // Same as "kNoPrefix" in "inference.cc".
  options.file_prefix = (options.file_prefix === undefined) ?
      '__NO_PREFIX__' :
      options.file_prefix;


  // Create model directory in RAM.
  const modelPath = 'model_' + Math.floor(Math.random() * 0xFFFFFFFF);
  Module.FS.mkdirTree(modelPath);

  // Unzip Model
  const zippedModel = await JSZip.loadAsync(serializedModel);

  // Extract model
  const promiseUncompressed = [];

  zippedModel.forEach((filename, file) => {
    promiseUncompressed.push(
        file.async('blob').then((data) => blobToArrayBuffer(data)).then((data) => {
          if (filename.endsWith('/')) {
            throw new Error(
                'The model zipfile is expected to be a flat zip file, but it contains a sub-directory. If zipping the model manually with the `zip` tool, make sure to use the `-j` option.');
          }
          Module.FS.writeFile(
              modelPath + '/' + filename, new Uint8Array(data),
              {'encoding': 'binary'});
        }));
  });

  await Promise.all(promiseUncompressed);

  // Load model in Yggdrasil.
  const modelWasm = Module.InternalLoadModel(
      modelPath, options.createdTFDFSignature, options.file_prefix);

  // Delete the model on disk.
  for (const filename of Module.FS.readdir(modelPath)) {
    if (filename === '.' || filename === '..') {
      continue;
    }
    Module.FS.unlink(modelPath + '/' + filename);
  }
  Module.FS.rmdir(modelPath);

  if (modelWasm == null) {
    throw new Error('Cannot parse model');
  }

  return new Model(modelWasm, options.createdTFDFSignature);
};

/**
 * Loads a model from a URL.
 *
 * Usage example:
 *
 *    let model = null;
 *    ydf.loadModelFromUrl("model.zip").then((loadedModel) => {
 *        model = loadedModel;
 *    }
 *
 * @param {string} url Url to a model.
 * @param {!LoadModelOptions=} options Loading model options.
 * @return {!Promise<!Model>} The loaded model.
 */
Module['loadModelFromUrl'] =
    async function loadModelFromUrl(url, options = undefined) {
  // Download model
  const serializedModel = await fetch(url).then((r) => r.blob());

  return Module['loadModelFromZipBlob'](serializedModel, options);
};
