/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as blazeface from '@tensorflow-models/blazeface';
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {Box, cutBoxFromImageAndResize, enlargeBox, getBoxCenter, getBoxSize, scaleBoxCoordinates, squarifyBox} from './box';
import {MESH_ANNOTATIONS} from './keypoints';
import {buildRotationMatrix, computeRotation, computeEyeRotation, Coord2D, Coord3D, Coords3D, dot, IDENTITY_MATRIX, invertTransformMatrix, rotatePoint, TransformationMatrix} from './util';

export type Prediction = {
  coords: tf.Tensor2D,        // coordinates of facial landmarks.
  scaledCoords: tf.Tensor2D,  // coordinates normalized to the mesh size.
  box: Box,                   // bounding box of coordinates.
  flag: tf.Scalar             // confidence in presence of a face.
};

const LANDMARKS_COUNT = 468;
const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.25;

const MESH_MOUTH_INDEX = 13;
const MESH_KEYPOINTS_LINE_OF_SYMMETRY_INDICES =
    [MESH_MOUTH_INDEX, MESH_ANNOTATIONS['midwayBetweenEyes'][0]];

const BLAZEFACE_MOUTH_INDEX = 3;
const BLAZEFACE_NOSE_INDEX = 2;
const BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES =
    [BLAZEFACE_MOUTH_INDEX, BLAZEFACE_NOSE_INDEX];

const LEFT_EYE_OUTLINE = MESH_ANNOTATIONS['leftEyeLower0'];
const LEFT_EYE_BOUNDS =
    [LEFT_EYE_OUTLINE[0], LEFT_EYE_OUTLINE[LEFT_EYE_OUTLINE.length - 1]];
const RIGHT_EYE_OUTLINE = MESH_ANNOTATIONS['rightEyeLower0'];
const RIGHT_EYE_BOUNDS =
    [RIGHT_EYE_OUTLINE[0], RIGHT_EYE_OUTLINE[RIGHT_EYE_OUTLINE.length - 1]];

const IRIS_UPPER_CENTER_INDEX = 3;
const IRIS_LOWER_CENTER_INDEX = 4;
const IRIS_IRIS_INDEX = 71;
const IRIS_NUM_COORDINATES = 76;

// Factor by which to enlarge the box around the eye landmarks so the input
// region matches the expectations of the iris model.

// TODO: confirm why this is 2.3 - I would expect 1.5 as per the model card
// (https://drive.google.com/file/d/1bsWbokp9AklH2ANjCfmjqEzzxO1CNbMu/view)
const ENLARGE_EYE_RATIO = 2.3;
const IRIS_MODEL_INPUT_SIZE = 64;

// A mapping from facemesh model keypoints to iris model keypoints.
const MESH_TO_IRIS_INDICES_MAP = [
  {key: 'EyeUpper0', indices: [9, 10, 11, 12, 13, 14, 15]},
  {key: 'EyeUpper1', indices: [25, 26, 27, 28, 29, 30, 31]},
  {key: 'EyeUpper2', indices: [41, 42, 43, 44, 45, 46, 47]},
  {key: 'EyeLower0', indices: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
  {key: 'EyeLower1', indices: [16, 17, 18, 19, 20, 21, 22, 23, 24]},
  {key: 'EyeLower2', indices: [32, 33, 34, 35, 36, 37, 38, 39, 40]},
  {key: 'EyeLower3', indices: [54, 55, 56, 57, 58, 59, 60, 61, 62]},
  {key: 'EyebrowUpper', indices: [63, 64, 65, 66, 67, 68, 69, 70]},
  {key: 'EyebrowLower', indices: [48, 49, 50, 51, 52, 53]}
];

// Replace the transformed coordinates returned by facemesh with refined iris model
// coordinates.
// Update the z coordinate to be an average of the original and the new. This
// produces the best visual effect.
function replaceCoordinates(
    coords: Coords3D, newCoords: Coords3D, prefix: string, keys?: string[]) {
  for (let i = 0; i < MESH_TO_IRIS_INDICES_MAP.length; i++) {
    const {key, indices} = MESH_TO_IRIS_INDICES_MAP[i];
    const originalIndices = MESH_ANNOTATIONS[`${prefix}${key}`];

    const shouldReplaceAllKeys = keys == null;
    if (shouldReplaceAllKeys || keys.includes(key)) {
      for (let j = 0; j < indices.length; j++) {
        const index = indices[j];

        coords[originalIndices[j]] = [
          newCoords[index][0], newCoords[index][1],
          (newCoords[index][2] + coords[originalIndices[j]][2]) / 2
        ];
      }
    }
  }
}

// The Pipeline coordinates between the bounding box and skeleton models.
export class Pipeline {
  // MediaPipe model for detecting facial bounding boxes.
  private boundingBoxDetector: blazeface.BlazeFaceModel;
  // MediaPipe model for detecting facial mesh.
  private meshDetector: tfconv.GraphModel;

  private meshWidth: number;
  private meshHeight: number;
  private maxContinuousChecks: number;
  private maxFaces: number;

  public irisModel: tfconv.GraphModel|null;

  // An array of facial bounding boxes.
  private regionsOfInterest: Box[] = [];
  private runsWithoutFaceDetector = 0;

  constructor(
      boundingBoxDetector: blazeface.BlazeFaceModel,
      meshDetector: tfconv.GraphModel, meshWidth: number, meshHeight: number,
      maxContinuousChecks: number, maxFaces: number,
      irisModel: tfconv.GraphModel|null) {
    this.boundingBoxDetector = boundingBoxDetector;
    this.meshDetector = meshDetector;
    this.irisModel = irisModel;
    this.meshWidth = meshWidth;
    this.meshHeight = meshHeight;
    this.maxContinuousChecks = maxContinuousChecks;
    this.maxFaces = maxFaces;
  }

  transformRawCoords(
      rawCoords: Coords3D, box: Box, angle: number,
      rotationMatrix: TransformationMatrix) {
    const boxSize =
        getBoxSize({startPoint: box.startPoint, endPoint: box.endPoint});
    const scaleFactor =
        [boxSize[0] / this.meshWidth, boxSize[1] / this.meshHeight];
    const coordsScaled = rawCoords.map(
        coord => ([
          scaleFactor[0] * (coord[0] - this.meshWidth / 2),
          scaleFactor[1] * (coord[1] - this.meshHeight / 2),
          // scale the z-coordinate here (based on face width)
          scaleFactor[0] * coord[2]
        ]));

    const coordsRotationMatrix = buildRotationMatrix(angle, [0, 0]);
    const coordsRotated = coordsScaled.map(
        (coord: Coord3D) =>
            ([...rotatePoint(coord, coordsRotationMatrix), coord[2]]));

    const inverseRotationMatrix = invertTransformMatrix(rotationMatrix);
    const boxCenter = [
      ...getBoxCenter({startPoint: box.startPoint, endPoint: box.endPoint}), 1
    ];

    const originalBoxCenter = [
      dot(boxCenter, inverseRotationMatrix[0]),
      dot(boxCenter, inverseRotationMatrix[1])
    ];

    return coordsRotated.map((coord): Coord3D => ([
                               coord[0] + originalBoxCenter[0],
                               coord[1] + originalBoxCenter[1], coord[2]
                             ]));
  }

  private getLeftToRightEyeDepthDifference(rawCoords: Coords3D): number {
    const leftEyeZ = rawCoords[LEFT_EYE_BOUNDS[0]][2];
    const rightEyeZ = rawCoords[RIGHT_EYE_BOUNDS[0]][2];
    return leftEyeZ - rightEyeZ;
  }

  // Returns a box describing a cropped region around the eye fit for passing to
  // the iris model.
  getEyeBox(
    coords: Coords3D, face: tf.Tensor4D, eyeInnerCornerIndex: number,
    eyeOuterCornerIndex: number,
    flip = false): { box: Box, boxSize: [number, number],
      angle: number, rotationMatrix: TransformationMatrix, crop: tf.Tensor4D } {

    const angle = computeEyeRotation(
      coords[eyeInnerCornerIndex], coords[eyeOuterCornerIndex], flip);

    const angleCompentationFactor = 1 / Math.max(
      Math.abs(Math.cos(angle)), Math.abs(Math.sin(angle))
    );

    const box = squarifyBox(enlargeBox(
      this.calculateLandmarksBoundingBox(
        [coords[eyeInnerCornerIndex], coords[eyeOuterCornerIndex]]),
      angleCompentationFactor * ENLARGE_EYE_RATIO));
    const boxSize = getBoxSize(box);

    const h = face.shape[1];
    const w = face.shape[2];

    const eyeCenter =
        getBoxCenter({startPoint: box.startPoint, endPoint: box.endPoint});
    const eyeCenterNormalized: Coord2D = [eyeCenter[0] / w, eyeCenter[1] / h];

    let rotatedImage = face;
    let rotationMatrix = IDENTITY_MATRIX;
    if (angle !== 0) {
      rotatedImage =
          tf.image.rotateWithOffset(face, angle, 0, eyeCenterNormalized);
      rotationMatrix = buildRotationMatrix(-angle, eyeCenter);
    }

    let crop = tf.image.cropAndResize(
      rotatedImage, [[
        box.startPoint[1] / h,
        box.startPoint[0] / w, box.endPoint[1] / h,
        box.endPoint[0] / w
      ]],
      [0], [IRIS_MODEL_INPUT_SIZE, IRIS_MODEL_INPUT_SIZE]);
    if (flip) {
      crop = tf.image.flipLeftRight(crop);
    }

    return { box, boxSize, angle, rotationMatrix, crop };
  }

  // Given a cropped image of an eye, returns the coordinates of the contours
  // surrounding the eye and the iris.
  getEyeCoords(
      eyeData: Float32Array, eyeBox: Box, eyeBoxSize: [number, number],
      angle: number, rotationMatrix: TransformationMatrix, flip = false): { coords: Coords3D, iris: Coords3D } {
    const eyeCoords: Coords3D = [];
    for (let i = 0; i < IRIS_NUM_COORDINATES; i++) {
      const x = eyeData[i * 3];
      const y = eyeData[i * 3 + 1];
      const z = eyeData[i * 3 + 2];
      eyeCoords.push([(flip ? (IRIS_MODEL_INPUT_SIZE - x) : x), y, z]);
    }

    const scaleFactor =
    [eyeBoxSize[0] / IRIS_MODEL_INPUT_SIZE, eyeBoxSize[1] / IRIS_MODEL_INPUT_SIZE];

    const eyeCoordsScaled = eyeCoords.map(
      coord => ([
        scaleFactor[0] * (coord[0] - IRIS_MODEL_INPUT_SIZE / 2),
        scaleFactor[1] * (coord[1] - IRIS_MODEL_INPUT_SIZE / 2),
        // scale the z-coordinate here...
        scaleFactor[0] * coord[2]
      ])
    );

    const eyeCoordsRotationMatrix = buildRotationMatrix(angle, [0, 0]);
    const eyeCoordsRotated = eyeCoordsScaled.map(
      (coord: Coord3D) =>
        ([...rotatePoint(coord, eyeCoordsRotationMatrix), coord[2]]));

    const inverseRotationMatrix = invertTransformMatrix(rotationMatrix);
    const eyeBoxCenter = [
      ...getBoxCenter({ startPoint: eyeBox.startPoint, endPoint: eyeBox.endPoint }), 1
    ];

    const originalEyeBoxCenter = [
      dot(eyeBoxCenter, inverseRotationMatrix[0]),
      dot(eyeBoxCenter, inverseRotationMatrix[1])
    ];

    const eyeCoordsTransformed = eyeCoordsRotated.map((coord): Coord3D => ([
      coord[0] + originalEyeBoxCenter[0],
      coord[1] + originalEyeBoxCenter[1], coord[2]
    ]));

    return {
      coords: eyeCoordsTransformed,
      iris: eyeCoordsTransformed.slice(IRIS_IRIS_INDEX)
    };
  }

  /**
   * Returns an array of predictions for each face in the input.
   * @param input - tensor of shape [1, H, W, 3].
   * @param predictIrises - Whether to return keypoints for the irises.
   */
  async predict(input: tf.Tensor4D, predictIrises: boolean):
      Promise<Prediction[]> {
    if (this.shouldUpdateRegionsOfInterest()) {
      const returnTensors = false;
      const annotateFace = true;
      const {boxes, scaleFactor} =
          await this.boundingBoxDetector.getBoundingBoxes(
              input, returnTensors, annotateFace);

      if (boxes.length === 0) {
        this.regionsOfInterest = [];
        return null;
      }

      const scaledBoxes =
          boxes.map((prediction: blazeface.BlazeFacePrediction): Box => {
            const predictionBoxCPU = {
              startPoint: prediction.box.startPoint.squeeze().arraySync() as
                  Coord2D,
              endPoint: prediction.box.endPoint.squeeze().arraySync() as Coord2D
            };

            const scaledBox =
                scaleBoxCoordinates(predictionBoxCPU, scaleFactor as Coord2D);
            const enlargedBox = enlargeBox(scaledBox);
            return {
              ...enlargedBox,
              landmarks: prediction.landmarks.arraySync() as Coords3D
            };
          });

      boxes.forEach((box: {
                      startPoint: tf.Tensor2D,
                      startEndTensor: tf.Tensor2D,
                      endPoint: tf.Tensor2D
                    }) => {
        if (box != null && box.startPoint != null) {
          box.startEndTensor.dispose();
          box.startPoint.dispose();
          box.endPoint.dispose();
        }
      });

      this.updateRegionsOfInterest(scaledBoxes);
      this.runsWithoutFaceDetector = 0;
    } else {
      this.runsWithoutFaceDetector++;
    }

    return tf.tidy(() => {
      return this.regionsOfInterest.map((box, i) => {
        let angle = 0;
        // The facial bounding box landmarks could come either from blazeface
        // (if we are using a fresh box), or from the mesh model (if we are
        // reusing an old box).
        const boxLandmarksFromMeshModel =
            box.landmarks.length >= LANDMARKS_COUNT;
        let [indexOfMouth, indexOfForehead] =
            MESH_KEYPOINTS_LINE_OF_SYMMETRY_INDICES;

        if (boxLandmarksFromMeshModel === false) {
          [indexOfMouth, indexOfForehead] =
              BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES;
        }

        angle = computeRotation(
            box.landmarks[indexOfMouth], box.landmarks[indexOfForehead]);

        const faceCenter =
            getBoxCenter({startPoint: box.startPoint, endPoint: box.endPoint});
        const faceCenterNormalized: Coord2D =
            [faceCenter[0] / input.shape[2], faceCenter[1] / input.shape[1]];

        let rotatedImage = input;
        let rotationMatrix = IDENTITY_MATRIX;
        if (angle !== 0) {
          rotatedImage =
              tf.image.rotateWithOffset(input, angle, 0, faceCenterNormalized);
          rotationMatrix = buildRotationMatrix(-angle, faceCenter);
        }

        const boxCPU = {startPoint: box.startPoint, endPoint: box.endPoint};
        const face: tf.Tensor4D =
            cutBoxFromImageAndResize(boxCPU, rotatedImage, [
              this.meshHeight, this.meshWidth
            ]).div(255);

        // The first returned tensor represents facial contours, which are
        // included in the coordinates.
        const [, flag, coords] =
            this.meshDetector.predict(
                face) as [tf.Tensor, tf.Tensor2D, tf.Tensor2D];

        const coordsReshaped: tf.Tensor2D = tf.reshape(coords, [-1, 3]);
        let rawCoords = coordsReshaped.arraySync() as Coords3D;

        let transformedCoords =
          this.transformRawCoords(rawCoords, box, angle, rotationMatrix);

        // TODO: here we are flipping the left eye so that it looks like a right
        // eye - this is different to the Model Card
        // (https://drive.google.com/file/d/1bsWbokp9AklH2ANjCfmjqEzzxO1CNbMu/view)
        // but consistent with examples shown here:
        // https://google.github.io/mediapipe/solutions/iris.html
        // clarify which is correct
        if (predictIrises) {
          const {
            box: leftEyeBox,
            boxSize: leftEyeBoxSize,
            angle: leftEyeAngle,
            rotationMatrix: leftEyeRotationMatrix,
            crop: leftEyeCrop
          } =
              this.getEyeBox(
                  transformedCoords, input, LEFT_EYE_BOUNDS[0], LEFT_EYE_BOUNDS[1],
                  true);
          const {
            box: rightEyeBox,
            boxSize: rightEyeBoxSize,
            angle: rightEyeAngle,
            rotationMatrix: rightEyeRotationMatrix,
            crop: rightEyeCrop
          } =
              this.getEyeBox(
                  transformedCoords, input, RIGHT_EYE_BOUNDS[0], RIGHT_EYE_BOUNDS[1]);

          const eyePredictions =
              (this.irisModel.predict(
                  tf.concat([leftEyeCrop.div(255), rightEyeCrop.div(255)]))) as tf.Tensor4D;
          const eyePredictionsData = eyePredictions.dataSync() as Float32Array;

          const leftEyeData =
              eyePredictionsData.slice(0, IRIS_NUM_COORDINATES * 3);
          const {coords: leftEyeCoords, iris: leftIrisCoords} =
              this.getEyeCoords(
                leftEyeData, leftEyeBox, leftEyeBoxSize,
                leftEyeAngle, leftEyeRotationMatrix, true
              );

          const rightEyeData =
              eyePredictionsData.slice(IRIS_NUM_COORDINATES * 3);
          const {coords: rightEyeCoords, iris: rightIrisCoords} =
              this.getEyeCoords(
                rightEyeData, rightEyeBox, rightEyeBoxSize,
                rightEyeAngle, rightEyeRotationMatrix
              );

          const leftToRightEyeDepthDifference =
              this.getLeftToRightEyeDepthDifference(rawCoords);
          if (Math.abs(leftToRightEyeDepthDifference) <
              30) {  // User is looking straight ahead.
            replaceCoordinates(transformedCoords, leftEyeCoords, 'left');
            replaceCoordinates(transformedCoords, rightEyeCoords, 'right');
          } else if (leftToRightEyeDepthDifference < 1) {  // User is looking
                                                           // towards the
                                                           // right.
            // If the user is looking to the left or to the right, the iris
            // coordinates tend to diverge too much from the mesh coordinates
            // for them to be merged. So we only update a single contour line
            // above and below the eye.
            replaceCoordinates(
                transformedCoords, leftEyeCoords, 'left',
                ['EyeUpper0', 'EyeLower0']);
          } else {  // User is looking towards the left.
            replaceCoordinates(
                transformedCoords, rightEyeCoords, 'right',
                ['EyeUpper0', 'EyeLower0']);
          }

          const adjustedLeftIrisCoords =
              this.getAdjustedIrisCoords(transformedCoords, leftIrisCoords, 'left');
          const adjustedRightIrisCoords = this.getAdjustedIrisCoords(
              transformedCoords, rightIrisCoords, 'right');
          transformedCoords = transformedCoords.concat(adjustedLeftIrisCoords)
                          .concat(adjustedRightIrisCoords);
        }

        const landmarksBox = enlargeBox(
            this.calculateLandmarksBoundingBox(transformedCoords));
        this.regionsOfInterest[i] = {
          ...landmarksBox,
          landmarks: transformedCoords
        };

        const prediction: Prediction = {
          coords: tf.tensor2d(rawCoords, [rawCoords.length, 3]),
          scaledCoords: tf.tensor2d(transformedCoords),
          box: landmarksBox,
          flag: flag.squeeze()
        };

        return prediction;
      });
    });
  }

  // Updates regions of interest if the intersection over union between
  // the incoming and previous regions falls below a threshold.
  updateRegionsOfInterest(boxes: Box[]) {
    for (let i = 0; i < boxes.length; i++) {
      const box = boxes[i];
      const previousBox = this.regionsOfInterest[i];
      let iou = 0;

      if (previousBox && previousBox.startPoint) {
        const [boxStartX, boxStartY] = box.startPoint;
        const [boxEndX, boxEndY] = box.endPoint;
        const [previousBoxStartX, previousBoxStartY] = previousBox.startPoint;
        const [previousBoxEndX, previousBoxEndY] = previousBox.endPoint;

        const xStartMax = Math.max(boxStartX, previousBoxStartX);
        const yStartMax = Math.max(boxStartY, previousBoxStartY);
        const xEndMin = Math.min(boxEndX, previousBoxEndX);
        const yEndMin = Math.min(boxEndY, previousBoxEndY);

        const intersection = (xEndMin - xStartMax) * (yEndMin - yStartMax);
        const boxArea = (boxEndX - boxStartX) * (boxEndY - boxStartY);
        const previousBoxArea = (previousBoxEndX - previousBoxStartX) *
            (previousBoxEndY - boxStartY);
        iou = intersection / (boxArea + previousBoxArea - intersection);
      }

      if (iou < UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD) {
        this.regionsOfInterest[i] = box;
      }
    }

    this.regionsOfInterest = this.regionsOfInterest.slice(0, boxes.length);
  }

  clearRegionOfInterest(index: number) {
    if (this.regionsOfInterest[index] != null) {
      this.regionsOfInterest = [
        ...this.regionsOfInterest.slice(0, index),
        ...this.regionsOfInterest.slice(index + 1)
      ];
    }
  }

  shouldUpdateRegionsOfInterest(): boolean {
    const roisCount = this.regionsOfInterest.length;
    const noROIs = roisCount === 0;

    if (this.maxFaces === 1 || noROIs) {
      return noROIs;
    }

    return roisCount !== this.maxFaces &&
        this.runsWithoutFaceDetector >= this.maxContinuousChecks;
  }

  calculateLandmarksBoundingBox(landmarks: Coords3D): Box {
    const xs = landmarks.map(d => d[0]);
    const ys = landmarks.map(d => d[1]);

    const startPoint: Coord2D = [Math.min(...xs), Math.min(...ys)];
    const endPoint: Coord2D = [Math.max(...xs), Math.max(...ys)];
    return {startPoint, endPoint};
  }
}
