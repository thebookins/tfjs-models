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
import * as THREE from 'three';

import { TRIANGULATION } from './triangulation';
import { tensor2d } from '@tensorflow/tfjs-core';
import { TLSSocket } from 'tls';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;
const GREEN = '#32EEDB';
const RED = "#FF2C35";
const BLUE = "#0000FF";
const WHITE = "#FFFFFF";

var calculateMean = true;

var noseDepthEstimate = {
  angles: [],
  xydist: [],
  values: [],
  estimate: null
};

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

function distance(a, b) {
  // Distance between to 2D points, a and b
  return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

function distanceXYZ(a, b) {
  // Distance between to ND points, a and b
  var n, sum;
  sum = 0.0;
  for (n = 0; n < a.length; n++) {
    sum += Math.pow(a[n] - b[n], 2);
  }
  return Math.sqrt(sum);
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
  triangulateMesh: false,
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

function average(a, b) {
  avg = [];
  for (i = 0; i < a.length; i++) {
      avg.push(0.5 * (a[i] + b[i]));
  }
  return avg;
}

function crossProduct(a, b) {
  let aa = a.arraySync();
  let bb = b.arraySync();
  let x = aa[1] * bb[2] - aa[2] * bb[1];
  let y = aa[2] * bb[0] - aa[0] * bb[2];
  let z = aa[0] * bb[1] - aa[1] * bb[0];
  return tf.tensor1d([x, y, z], 'float32');
}

function radToDegrees(rad) {
  return rad * 180 / Math.PI;
}

function degToRadians(deg) {
  return deg * Math.PI / 180;
}

function headCsysCanonical() {
  let M = headCsysCanonical_threejs();
  console.log(M);
  // Coordinates from canonical_face_model.obj in mediapipe repo
  let p_tL = tf.tensor1d([ 7.66418, 0.673132, -2.43587], 'float32');
  let p_tR = tf.tensor1d([-7.66418, 0.673132, -2.43587], 'float32');
  let p_iL = tf.tensor1d([ 3.32732, 0.104863,  4.11386], 'float32');
  let p_iR = tf.tensor1d([-3.32732, 0.104863,  4.11386], 'float32'); 
  // p0 - origin
  let p0 = p_tL.stack(p_tR).mean(0);
  // p1 = point on x axis, v1 = x-axis
  let p1 = p_tL;
  let v1 = p1.sub(p0);
  let v1_norm = v1.div(v1.norm());
  // p2 = point on z axis, v3 = z-axis
  let p2 = p_iL.stack(p_iR).mean(0);
  let v3 = p2.sub(p0);
  let v3_norm = v3.div(v3.norm());
  // v2 = y-axis
  let v2 = crossProduct(v3_norm, v1_norm);
  let v2_norm = v2.div(v2.norm());
  // Recalculate v1 to ensure that csys is orthogonal
  v1 = crossProduct(v2_norm, v3_norm);
  v1_norm = v1.div(v1.norm());
  // Clean-up
  tf.dispose(p_tL, p_tR, p_iL, p_iR, p0, p1, p2, v1, v2, v3);
  // Return matrix representing csys
  tf.stack([v1_norm,v2_norm,v3_norm]).print();
  return tf.stack([v1_norm,v2_norm,v3_norm]);
}

function headCsysFromPoints(ptL, ptR, piL, piR) {
  // p0 - origin
  let p0 = ptL.clone().lerp(ptR, 0.5);
  // p1 = point on x axis, v1 = x-axis
  let p1 = ptL.clone();
  let v1 = new THREE.Vector3().subVectors(p1,p0).normalize();
  // p2 = point on z axis, v3 = z-axis
  let p2 = piL.clone().lerp(piR, 0.5);
  let v3 = new THREE.Vector3().subVectors(p2,p0).normalize();
  // v2 = y-axis
  let v2 = new THREE.Vector3().crossVectors(v3, v1).normalize();
  // Recalculate v1 to ensure that csys is orthogonal
  v1.crossVectors(v2, v3).normalize();
  // Clean-up
  // Return matrix representing csys
  let basis = new THREE.Matrix4().makeBasis(v1, v2, v3);
  return new THREE.Matrix3().setFromMatrix4(basis);
}

function headCsysMoving_threejs(mesh) {
  // Landmarks used to create coordinate system
  const LMRK = {
    TRAGION_L:  454,
    TRAGION_R:  234,
    INFRAORB_L: 330,
    INFRAORB_R: 101
  };
  // Landmark coordinates
  let ptL = new THREE.Vector3().fromArray(mesh[LMRK.TRAGION_L]);
  let ptR = new THREE.Vector3().fromArray(mesh[LMRK.TRAGION_R]);
  let piL = new THREE.Vector3().fromArray(mesh[LMRK.INFRAORB_L]);
  let piR = new THREE.Vector3().fromArray(mesh[LMRK.INFRAORB_R]);
  // Basis matrix
  return headCsysFromPoints(ptL, ptR, piL, piR);
}

function headCsysCanonical_threejs() {
  // Coordinates from canonical_face_model.obj in mediapipe repo
  let ptL = new THREE.Vector3( 7.66418, 0.673132, -2.43587);
  let ptR = new THREE.Vector3(-7.66418, 0.673132, -2.43587);
  let piL = new THREE.Vector3( 3.32732, 0.104863,  4.11386);
  let piR = new THREE.Vector3(-3.32732, 0.104863,  4.11386);
  // Basis matrix
  return headCsysFromPoints(ptL, ptR, piL, piR);  
}

  // p0 - origin
  //let p0 = p_tL.clone();
  //p0.lerp(p_tR, 0.5);
  // p1 = point on x axis, v1 = x-axis
  //let p1 = p_tL.clone();
  //let v1_norm = new THREE.Vector3();
  //v1_norm.subVectors(p1,p0).normalize();
  // p2 = point on z axis, v3 = z-axis
  //let p2 = p_iL.clone();
  //p2.lerp(p_iR, 0.5);
  //let v3_norm = new THREE.Vector3();
  //v3_norm.subVectors(p2,p0).normalize();
  // v2 = y-axis
  //let v2_norm = new THREE.Vector3();
  //v2_norm.crossVectors(v3_norm, v1_norm).normalize();
  // Recalculate v1 to ensure that csys is orthogonal
  //v1_norm.crossVectors(v2_norm, v3_norm).normalize();
  // Clean-up
  // Return matrix representing csys
  //let basis = new THREE.Matrix4();
  //basis.makeBasis(v1_norm, v2_norm, v3_norm);
  //let R = new THREE.Matrix3();
  //R.setFromMatrix4(basis);
  //console.log(R);
//}

function headCsys(mesh) {
  let M = headCsysMoving_threejs(mesh);
  console.log(M);
  // Landmarks used to create coordinate system
  const LM_TRAGIONL  = 454;
  const LM_TRAGIONR  = 234;
  const LM_INFRAORBL = 330; 
  const LM_INFRAORBR = 101;
  // Landmark coordinates
  let p_tL = tf.tensor1d(mesh[LM_TRAGIONL],  'float32');
  let p_tR = tf.tensor1d(mesh[LM_TRAGIONR],  'float32');
  let p_iL = tf.tensor1d(mesh[LM_INFRAORBL], 'float32');
  let p_iR = tf.tensor1d(mesh[LM_INFRAORBR], 'float32');
  // Visualise additional landmarks
  ctx.fillStyle = BLUE;
  ctx.strokeStyle = BLUE;
  ctx.lineWidth = 1;  
  ctx.beginPath();
  ctx.arc(mesh[LM_INFRAORBL][0], mesh[LM_INFRAORBL][1], 3 /* radius */, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(mesh[LM_INFRAORBR][0], mesh[LM_INFRAORBR][1], 3 /* radius */, 0, 2 * Math.PI);
  ctx.stroke();
  // p0 = origin
  let p0 = p_tL.stack(p_tR).mean(0);
  // p1 = point on x axis, v1 = x-axis
  let p1 = p_tL;
  let v1 = p1.sub(p0);
  let v1_norm = v1.div(v1.norm());
  // p2 = point on z axis, v3 = z-axis
  let p2 = p_iL.stack(p_iR).mean(0);
  let v3 = p2.sub(p0);
  let v3_norm = v3.div(v3.norm());
  // v2 = y-axis
  let v2 = crossProduct(v3_norm, v1_norm);
  let v2_norm = v2.div(v2.norm());
  // Recalculate v1 to ensure that csys is orthogonal
  v1 = crossProduct(v2_norm, v3_norm);
  v1_norm = v1.div(v1.norm());
  // Clean-up
  tf.dispose(p_tL, p_tR, p_iL, p_iR, p0, p1, p2, v1, v2, v3);
  // Return matrix representing csys
  tf.stack([v1_norm,v2_norm,v3_norm]).print();
  return tf.stack([v1_norm,v2_norm,v3_norm]);
}

function estimateNoseDepth(mesh, irisScaleFactor) {
  // Landmark list
  const LM_NOSE_T = 4;
  const LM_NOSE_L = 278;
  const LM_NOSE_R = 48;
  // Landmark coordinates
  let p_nt = mesh[LM_NOSE_T];
  let p_nl = tf.tensor1d(mesh[LM_NOSE_L], 'float32');
  let p_nr = tf.tensor1d(mesh[LM_NOSE_R], 'float32');
  let p_nm = p_nl.stack(p_nr).mean(0).arraySync();
  // Get xy distance between points. Apply irisScaleFactor to remove scaling effect
  // when user moves closer / further away from the camera
  let xy = distance(p_nt, p_nm) * irisScaleFactor;
  //console.log(xy);
  return xy;
}

function headPose(mesh) {
      // Head coordinate system - For head post
      let M = headCsys(mesh).arraySync();
      let G = headCsysCanonical().arraySync();
      //let gloX = tf.tensor1d([1.0,0,0], 'float32');
      //let gloY = tf.tensor1d([0,1.0,0], 'float32');
      //let gloZ = tf.tensor1d([0,0,1.0], 'float32');
      let gloX = tf.tensor1d(G[0], 'float32');
      let gloY = tf.tensor1d(G[1], 'float32');
      let gloZ = tf.tensor1d(G[2], 'float32');
      let locZ = tf.tensor1d(M[2], 'float32');
      var poseLR  = radToDegrees(tf.asin(tf.dot(gloX, locZ)).arraySync());
      let poseUD  = radToDegrees(tf.asin(tf.dot(gloY, locZ)).arraySync());
      //let poseROT = radToDegrees(tf.asin(tf.dot(gloY, locX)).arraySync());
      let angleLR  = Math.abs(poseLR);
      let facingLR = poseLR >= 0.0 ? 'LEFT' : 'RIGHT';
      let angleUD  = Math.abs(poseUD);
      let facingUD = poseUD <= 0.0 ? 'UPWARDS' : 'DOWNWARDS';
      let headPosition = "Head position: " + angleLR.toFixed(1) + " deg to the " + facingLR;
      headPosition += " and " + angleUD.toFixed(1) + " deg " + facingUD;
      document.getElementById('head-pose').innerHTML = headPosition;
      // return
      return [poseLR, poseUD]
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
  const landmark_infraorbR = 101;
  const landmark_infraorbL = 330;  

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
  var irisScaleFactor = null;
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
    const cf = 1.1;
    const irisDiameter = 0.5 * (leftDiameter + rightDiameter);
    var irisScaleFactor = cf * 11.7 / irisDiameter;

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

  return irisScaleFactor;

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

      // Head pose
      let hp = headPose(prediction.scaledMesh);
      let poseLR = hp[0];
      let poseUD = hp[1];

      // Plots landmarks and update landmark measurements
      let irisScaleFactor = getLandmarkMeasurements(prediction);
      if (irisScaleFactor != null) {

        // Estimate nose depth
        let angleLR = Math.abs(poseLR);
        let angleUD = Math.abs(poseUD);
        if ((angleLR >= 10.0) && (angleLR <= 30.0) && (angleUD <= 10.0)) {

          let xyDistance = estimateNoseDepth(prediction.scaledMesh, irisScaleFactor);

          if (noseDepthEstimate.values.length < 50) {
            let nd = xyDistance / Math.sin(degToRadians(angleLR));
            noseDepthEstimate.values.push(nd);
            noseDepthEstimate.angles.push(angleLR);
            noseDepthEstimate.xydist.push(xyDistance);
          } else if ((noseDepthEstimate.values.length == 50) && (calculateMean)) {
            // NOTE: Probably better to replace this with a linear regression of angles vs xydist, then 
            //       extrapolate this curve to a value of angle = 90 degrees. 
            // Also need a way of accounting for length due to angleUD. This will particularly influence 
            // the values of xyDistance at low angleLR values
            // BETTER YET: Also include angleUD in the regression, and then find the values of angleLR and
            //             angleUD such that length is 0 (call this angleUD = alpha). Then extrapolate to 
            //             the nose depth at angleLR = 90 deg and angleUD = alpha deg ie. no contribution 
            //             from angleUD
            noseDepthEstimate.estimate = tf.tensor1d(noseDepthEstimate.values, 'float32').mean(0).arraySync();
            console.log(noseDepthEstimate);
            calculateMean = false;
          }
        }
      }
      
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
