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
  MainModule,
  vectorFloat,
  vectorString,
} from '../training_for_types';

declare var Module: MainModule;

/**
 * Converts a JavaScript string array to a C++ vector of EmbindString.
 * The vector must be manually freed after use to avoid memory leaks.
 */
export function stringArrayToVectorString(src: string[]): vectorString {
  const vector = Module.CreateVectorString(src.length);
  for (const value of src) {
    vector.push_back(value as EmbindString);
  }
  return vector;
}

/**
 * Converts a JavaScript number array to a C++ vector of float.
 * The vector must be manually freed after use to avoid memory leaks.
 */
export function numberArrayToVectorFloat(src: number[]): vectorFloat {
  const vector = Module.CreateVectorFloat(src.length);
  for (const value of src) {
    vector.push_back(value);
  }
  return vector;
}

/**
 * Converts a C++ EmbindString to a JavaScript string.
 */
export function embindStringToString(
  src: EmbindString,
  decoder: TextDecoder,
): string {
  if (typeof src === 'string') {
    return src;
  } else if (src instanceof ArrayBuffer) {
    return decoder.decode(src);
  } else {
    // Uint8Array, Uint8ClampedArray, or Int8Array
    return decoder.decode(src.buffer);
  }
}

/**
 * Converts a C++ vector of EmbindString to a JavaScript string array.
 */
export function ccVectorStringToJSVector(src: vectorString): string[] {
  const res: string[] = [];
  const decoder = new TextDecoder('utf-8');
  for (let i = 0; i < src.size(); i++) {
    const item = src.get(i);
    if (item === undefined) {
      throw new Error(`Found undefined element at index ${i}`);
    }
    res.push(embindStringToString(item, decoder));
  }
  return res;
}

/**
 * Converts a C++ vector of float to a JavaScript number array.
 */
export function ccVectorFloatToJSVector(src: vectorFloat): number[] {
  const res: number[] = [];
  for (let i = 0; i < src.size(); i++) {
    const item = src.get(i);
    if (item === undefined) {
      throw new Error(`Found undefined element at index ${i}`);
    }
    res.push(item);
  }
  return res;
}

/**
 * Converts a Blob into a promise resolving to an arrayBuffer.
 */
export function blobToArrayBuffer(blob: Blob) {
  if (blob.arrayBuffer) {
    return blob.arrayBuffer();
  } else {
    return new Promise((resolve) => {
      const fileReader = new FileReader();

      fileReader.readAsArrayBuffer(blob);
      fileReader.onload = (event) => {
        if (event.target !== null) {
          resolve(event.target.result);
        } else {
          throw new Error('Internal Error');
        }
      };
    });
  }
}
