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

import * as facemesh from '@tensorflow-models/facemesh';
import Stats from 'stats.js';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

import { TRIANGULATION } from './triangulation';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;
const GREEN = '#32EEDB';
const RED = "#FF2C35";
const BLUE = "#0000FF";
const WHITE = "#FFFFFF";
const BLACK = "#000000";
const GRAY = "#505050";


function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

function distance(a, b) {
  return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

let model, ctx, videoWidth, videoHeight, video, canvas,
  scatterGLHasInitialized = false, scatterGL, rafID;

const VIDEO_SIZE = 500;
const mobile = isMobile();
// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.
//const renderPointcloud = mobile === false;
const renderPointcloud = false; // Turn off for testing
const stats = new Stats();
const state = {
  backend: 'webgl',
  maxFaces: 1,
  triangulateMesh: true,
  predictIrises: true
};

if (renderPointcloud) {
  state.renderPointcloud = true;
}

function setupDatGui() {
  const gui = new dat.GUI();
  gui.add(state, 'backend', ['webgl', 'wasm', 'cpu'])
    .onChange(async backend => {
      window.cancelAnimationFrame(rafID);
      await tf.setBackend(backend);
      requestAnimationFrame(renderPrediction);
    });

  gui.add(state, 'maxFaces', 1, 20, 1).onChange(async val => {
    model = await facemesh.load({ maxFaces: val });
  });

  gui.add(state, 'triangulateMesh');
  gui.add(state, 'predictIrises');

  if (renderPointcloud) {
    gui.add(state, 'renderPointcloud').onChange(render => {
      document.querySelector('#scatter-gl-container').style.display =
        render ? 'inline-block' : 'none';
    });
  }
}

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width: mobile ? undefined : VIDEO_SIZE,
      height: mobile ? undefined : VIDEO_SIZE
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

function getLandmarkMeasurements(prediction) {

  //console.log(prediction);

  const mesh = prediction.mesh;
  const scaledMesh = prediction.scaledMesh;
  let x, y;

  // Landmark list
  const landmark_faceL = 454;
  const landmark_faceR = 234;
  const landmark_noseL = 278;
  const landmark_noseR = 48;
  const landmark_noseTip = 4;
  const landmark_sellion = 168;
  const landmark_supramenton = 200;

  // Face height
  const faceHeight = distance(
    mesh[landmark_sellion],
    mesh[landmark_supramenton]);

  const faceHeightScaled = distance(
    scaledMesh[landmark_sellion],
    scaledMesh[landmark_supramenton]);

  ctx.fillStyle = BLUE;
  ctx.strokeStyle = BLUE;
  ctx.lineWidth = 1;
  x = scaledMesh[landmark_sellion][0];
  y = scaledMesh[landmark_sellion][1];
  ctx.beginPath();
  ctx.arc(x, y, 3 /* radius */, 0, 2 * Math.PI);
  ctx.fill();
  x = scaledMesh[landmark_supramenton][0];
  y = scaledMesh[landmark_supramenton][1];
  ctx.beginPath();
  ctx.arc(x, y, 3 /* radius */, 0, 2 * Math.PI);
  ctx.fill();

  // Face width
  const faceWidth = distance(
    mesh[landmark_faceL],
    mesh[landmark_faceR]);

  const faceWidthScaled = distance(
    scaledMesh[landmark_faceL],
    scaledMesh[landmark_faceR]);

  ctx.fillStyle = RED;
  ctx.strokeStyle = RED;
  ctx.lineWidth = 1;
  x = scaledMesh[landmark_faceL][0];
  y = scaledMesh[landmark_faceL][1];
  ctx.beginPath();
  ctx.arc(x, y, 3 /* radius */, 0, 2 * Math.PI);
  ctx.fill();
  x = scaledMesh[landmark_faceR][0];
  y = scaledMesh[landmark_faceR][1];
  ctx.beginPath();
  ctx.arc(x, y, 3 /* radius */, 0, 2 * Math.PI);
  ctx.fill();

  // Nose width
  const noseWidth = distance(
    mesh[landmark_noseL],
    mesh[landmark_noseR]);

  const noseWidthScaled = distance(
    scaledMesh[landmark_noseL],
    scaledMesh[landmark_noseR]);

  ctx.fillStyle = WHITE;
  ctx.strokeStyle = WHITE;
  ctx.lineWidth = 1;
  x = scaledMesh[landmark_noseL][0];
  y = scaledMesh[landmark_noseL][1];
  ctx.beginPath();
  ctx.arc(x, y, 3 /* radius */, 0, 2 * Math.PI);
  ctx.fill();
  x = scaledMesh[landmark_noseR][0];
  y = scaledMesh[landmark_noseR][1];
  ctx.beginPath();
  ctx.arc(x, y, 3 /* radius */, 0, 2 * Math.PI);
  ctx.fill();

  // Nose depth
  const noseDepthL = distance(
    mesh[landmark_noseL],
    mesh[landmark_noseTip]);
  const noseDepthR = distance(
    mesh[landmark_noseR],
    mesh[landmark_noseTip]);
  const noseDepth = 0.5 * (noseDepthL + noseDepthR);

  const noseDepthScaledL = distance(
    scaledMesh[landmark_noseL],
    scaledMesh[landmark_noseTip]);
  const noseDepthScaledR = distance(
    scaledMesh[landmark_noseR],
    scaledMesh[landmark_noseTip]);
  const noseDepthScaled = 0.5 * (noseDepthScaledL + noseDepthScaledR);

  ctx.fillStyle = WHITE;
  ctx.strokeStyle = WHITE;
  ctx.lineWidth = 1;
  x = scaledMesh[landmark_noseTip][0];
  y = scaledMesh[landmark_noseTip][1];
  ctx.beginPath();
  ctx.arc(x, y, 3 /* radius */, 0, 2 * Math.PI);
  ctx.fill();

  // Iris model
  if (scaledMesh.length > NUM_KEYPOINTS) {

    ctx.fillStyle = RED;
    ctx.strokeStyle = RED;
    let i;
    for (i = 0; i < 10; i++) {
      x = scaledMesh[NUM_KEYPOINTS + i][0];
      y = scaledMesh[NUM_KEYPOINTS + i][1];
      ctx.beginPath();
      ctx.arc(x, y, 2 /* radius */, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Left iris
    const leftDiameterX = distance(
      scaledMesh[NUM_KEYPOINTS + 1],
      scaledMesh[NUM_KEYPOINTS + 3]);
    const leftDiameterY = distance(
      scaledMesh[NUM_KEYPOINTS + 2],
      scaledMesh[NUM_KEYPOINTS + 4]);
    const leftDiameter = 0.5 * (leftDiameterX + leftDiameterY);
    document.getElementById('measure-scaled-iris-diameter-L').innerHTML = "Scaled iris diameter L = " + leftDiameter.toFixed(1);

    // Right iris
    const rightDiameterX = distance(
      scaledMesh[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1],
      scaledMesh[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3]);
    const rightDiameterY = distance(
      scaledMesh[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
      scaledMesh[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]);
    const rightDiameter = 0.5 * (rightDiameterX + rightDiameterY);
    document.getElementById('measure-scaled-iris-diameter-R').innerHTML = "Scaled iris diameter R = " + rightDiameter.toFixed(1);

    // Scale all measurements using the iris diameter
    // Iris diameter should be 11.7 +- 0.5mm
    const irisDiameter = 0.5 * (leftDiameter + rightDiameter);
    const irisScaleFactor = 11.7 / irisDiameter;

    const fh = faceHeightScaled * irisScaleFactor;
    document.getElementById('measure-face-height').innerHTML = "Sellion-supramenton = " + fh.toFixed(1) + " mm";
    const fw = faceWidthScaled * irisScaleFactor;
    document.getElementById('measure-face-width').innerHTML = "Face width = " + fw.toFixed(1) + " mm";
    const nw = noseWidthScaled * irisScaleFactor;
    document.getElementById('measure-nose-width').innerHTML = "Nose width = " + nw.toFixed(1) + " mm";
    const nd = noseDepthScaled * irisScaleFactor;
    document.getElementById('measure-nose-depth').innerHTML = "Nose depth = " + nd.toFixed(1) + " mm";

  } else {

    document.getElementById('measure-face-height').innerHTML = "Sellion-supramenton = " + faceHeight.toFixed(1) + " mm";
    document.getElementById('measure-face-width').innerHTML = "Face width = " + faceWidth.toFixed(1) + " mm";
    document.getElementById('measure-nose-width').innerHTML = "Nose width = " + noseWidth.toFixed(1) + " mm";
    document.getElementById('measure-nose-depth').innerHTML = "Nose depth = " + noseDepth.toFixed(1) + " mm";
    document.getElementById('measure-scaled-iris-diameter-L').innerHTML = "";
    document.getElementById('measure-scaled-iris-diameter-R').innerHTML = "";
  }

}

async function renderPrediction() {
  stats.begin();

  const predictions = await model.estimateFaces(
    video, false /* returnTensors */, false /* flipHorizontal */,
    state.predictIrises);
  ctx.drawImage(
    video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);

  if (predictions.length > 0) {

    predictions.forEach(prediction => {
      const keypoints = prediction.scaledMesh;

      if (state.triangulateMesh) {
        ctx.strokeStyle = GREEN;
        ctx.lineWidth = 0.5;

        for (let i = 0; i < TRIANGULATION.length / 3; i++) {
          const points = [
            TRIANGULATION[i * 3], TRIANGULATION[i * 3 + 1],
            TRIANGULATION[i * 3 + 2]
          ].map(index => keypoints[index]);

          drawPath(ctx, points, true);
        }
      } else {
        ctx.fillStyle = GREEN;

        for (let i = 0; i < NUM_KEYPOINTS; i++) {
          const x = keypoints[i][0];
          const y = keypoints[i][1];

          ctx.beginPath();
          ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }
      }

      if (keypoints.length > NUM_KEYPOINTS) {
        ctx.strokeStyle = RED;
        ctx.lineWidth = 1;

        const leftCenter = keypoints[NUM_KEYPOINTS];
        const leftDiameterY = distance(
          keypoints[NUM_KEYPOINTS + 4],
          keypoints[NUM_KEYPOINTS + 2]);
        const leftDiameterX = distance(
          keypoints[NUM_KEYPOINTS + 3],
          keypoints[NUM_KEYPOINTS + 1]);

        const leftDiameter = 0.5 * (leftDiameterX + leftDiameterY);
        ctx.beginPath();
        ctx.ellipse(leftCenter[0], leftCenter[1], leftDiameter / 2.0, leftDiameter / 2.0, 0, 0, 2 * Math.PI);
        ctx.stroke();

        if (keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
          const rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
          const rightDiameterY = distance(
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]);
          const rightDiameterX = distance(
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]);

          const rightDiameter = 0.5 * (rightDiameterX + rightDiameterY);
          ctx.beginPath();
          ctx.ellipse(rightCenter[0], rightCenter[1], rightDiameter / 2.0, rightDiameter / 2.0, 0, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
      // Plots landmarks and update landmark measurements
      getLandmarkMeasurements(prediction);

    });

    if (renderPointcloud && state.renderPointcloud && scatterGL != null) {
      const pointsData = predictions.map(prediction => {
        let scaledMesh = prediction.scaledMesh;
        return scaledMesh.map(point => ([-point[0], -point[1], -point[2]]));
      });

      let flattenedPointsData = [];
      for (let i = 0; i < pointsData.length; i++) {
        flattenedPointsData = flattenedPointsData.concat(pointsData[i]);
      }
      const dataset = new ScatterGL.Dataset(flattenedPointsData);

      if (!scatterGLHasInitialized) {
        scatterGL.setPointColorer((i) => {
          if (i >= NUM_KEYPOINTS) {
            return RED;
          }
          return BLUE;
        });
        scatterGL.render(dataset);
      } else {
        scatterGL.updateDataset(dataset);
      }
      scatterGLHasInitialized = true;
    }
  }

  stats.end();
  rafID = requestAnimationFrame(renderPrediction);
};

async function main() {
  await tf.setBackend(state.backend);
  setupDatGui();

  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);

  await setupCamera();
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  const canvasContainer = document.querySelector('.canvas-wrapper');
  canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

  ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.fillStyle = GREEN;
  ctx.strokeStyle = GREEN;
  ctx.lineWidth = 0.5;

  model = await facemesh.load({ maxFaces: state.maxFaces });
  renderPrediction();

  if (renderPointcloud) {
    document.querySelector('#scatter-gl-container').style =
      `width: ${VIDEO_SIZE}px; height: ${VIDEO_SIZE}px;`;

    scatterGL = new ScatterGL(
      document.querySelector('#scatter-gl-container'),
      { 'rotateOnStart': false, 'selectEnabled': false });
  }
};

main();
