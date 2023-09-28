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
 * @fileoverview Internal and external definitions.
 * @externs
 */

/**
 * A std::vector<T>.
 *
 * @typedef {{
 *   size: function(): number,
 *   get: function(number) : (string|boolean|number|!Object),
 *   push_back: function(?),
 *   delete: function(),
 * }}
 */
let CCVector;

/**
 * A std::vector<std::vector<T>>.
 *
 * @typedef {{
 *   size: function(): number,
 *   get: function(number) : !CCVector,
 *   delete: function(),
 * }}
 */
let CCVectorVector;

/**
 * Structure of the InputFeature defined in "inference.cc".
 *
 * @typedef {{
 *   name: string,
 *   type: string,
 *   internalIdx: number,
 *   specIdx: number,
 * }}
 */
let InputFeature;

/**
 * Structure of the vector of InputFeature defined in "inference.cc".
 *
 * @typedef {{
 *   size: function(): number,
 *   get: function(number) : !InputFeature,
 *   push_back: function(!InputFeature),
 *   delete: function(),
 * }}
 */
let InternalInputFeatures;

/**
 * Structure of the vector of float defined in "inference.cc".
 *
 * @typedef {{
 *   size: function(): number,
 *   get: function(number) : number,
 *   push_back: function(number),
 *   delete: function(),
 * }}
 */
let InternalPredictions;

/**
 * Internal predictions with a TF-DF signature.
 *
 * @typedef {{
 *   densePredictions: !CCVectorVector,
 *   denseColRepresentation: !CCVector,
 *   delete: function(),
 * }}
 */
let InternalTFDFPredictions;

/**
 * Option for loading a model.
 *
 * @typedef {{
 *   createdTFDFSignature: (boolean | undefined),
 *   file_prefix: (string | undefined),
 * }}
 */
let LoadModelOptions;

/**
 * Output predictions when using the TF-DF signature.
 *
 * @typedef {{
 *   densePredictions: !Array<!Array<number>>,
 *   denseColRepresentation: Array<string>,
 * }}
 */
let TFDFOutputPrediction;

/**
 * Input features for the TF-DF signature.
 *
 * @typedef {{
 *   numericalFeatures: !Array<!Array<number>>,
 *   booleanFeatures: !Array<!Array<number>>,
 *   categoricalIntFeatures: !Array<!Array<number>>,
 *   categoricalSetIntFeaturesValues: !Array<number>,
 *   categoricalSetIntFeaturesRowSplitsDim1: !Array<number>,
 *   categoricalSetIntFeaturesRowSplitsDim2: !Array<number>,
 *   denseOutputDim: number,
 * }}
 */
let TFDFInput;

/**
 * Structure of the InternalModel defined in "inference.cc".
 *
 * @typedef {{
 *   predict: function(): !InternalPredictions,
 *   predictTFDFSignature: function(number): !InternalTFDFPredictions,
 *   newBatchOfExamples: function(number),
 *   setNumerical: function(number,number,number),
 *   setBoolean: function(number,number,number),
 *   setCategoricalInt: function(number,number,number),
 *   setCategoricalString: function(number,number,string),
 *   setCategoricalSetString: function(number,number,!CCVector),
 *   setCategoricalSetInt: function(number,number,!CCVector),
 *   getInputFeatures: function(): !InternalInputFeatures,
 *   getProtoInputFeatures: function(): !InternalInputFeatures,
 *   delete: function(),
 * }}
 */
let InternalModel;


/**
 * emscripten generated module.
 *
 * @typedef {{
 *    FS : {
 *      mkdirTree: function(string),
 *      writeFile: function(string, ?, ?),
 *      readdir: function(string),
 *      unlink: function(string),
 *      rmdir: function(string),
 *      }
 * }}
 */
let Module;

/**
 * JSZip utility.
 *
 * See https://stuk.github.io/jszip/documentation/api_jszip/for_each.html
 *
 * @typedef {{
 *   loadAsync: function(string),
 * }}
 */
let JSZip;
