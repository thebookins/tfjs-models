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
//import * as THREE from 'https://unpkg.com/three@0.106.2/build/three.min.js';

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

// Landmark list
const LMRK = {
  infraorb_L: 330,
  infraorb_R: 101,
  nose_L: 278,
  nose_R: 48,
  nosetip: 4,
  sellion: 168,
  supramenton: 200,
  tragion_L: 454,
  tragion_R: 234
};

// Face dimension ranges
const noseDepthRange = {min: 10, max: 40}
const noseWidthRange = {min: 20, max: 50}

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

function headCsysFromPoints(ptL, ptR, piL, piR) {
  // p0 - origin
  let p0 = ptL.clone().lerp(ptR, 0.5);
  // p1 = point on x axis, v1 = x-axis
  let p1 = ptL.clone();
  let v1 = new THREE.Vector3().subVectors(p1, p0).normalize();
  // p2 = point on z axis, v3 = z-axis
  let p2 = piL.clone().lerp(piR, 0.5);
  let v3 = new THREE.Vector3().subVectors(p2, p0).normalize();
  // v2 = y-axis
  let v2 = new THREE.Vector3().crossVectors(v3, v1).normalize();
  // Recalculate v1 to ensure that csys is orthogonal
  v1.crossVectors(v2, v3).normalize();
  // Return matrix representing csys
  let basis = new THREE.Matrix4().makeBasis(v1, v2, v3);
  return new THREE.Matrix3().setFromMatrix4(basis);
}

function headCsysMoving(mesh) {
  // Landmark coordinates
  let ptL = new THREE.Vector3().fromArray(mesh[LMRK.tragion_L]);
  let ptR = new THREE.Vector3().fromArray(mesh[LMRK.tragion_R]);
  let piL = new THREE.Vector3().fromArray(mesh[LMRK.infraorb_L]);
  let piR = new THREE.Vector3().fromArray(mesh[LMRK.infraorb_R]);
  // Plot the landmarks
  plotLandmark(piL);
  plotLandmark(piR);
  // Basis matrix
  return headCsysFromPoints(ptL, ptR, piL, piR);
}

function headCsysCanonical() {
  // Coordinates from canonical_face_model.obj in mediapipe repo
  let ptL = new THREE.Vector3( 7.66418, 0.673132, -2.43587);
  let ptR = new THREE.Vector3(-7.66418, 0.673132, -2.43587);
  let piL = new THREE.Vector3( 3.32732, 0.104863,  4.11386);
  let piR = new THREE.Vector3(-3.32732, 0.104863,  4.11386);
  // Basis matrix
  return headCsysFromPoints(ptL, ptR, piL, piR);  
}

function plotLandmark(lmrk, colour=BLUE, radius=3, lineWidth=1) {
  ctx.fillStyle = colour;
  ctx.strokeStyle = colour;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.arc(lmrk.x, lmrk.y, radius, 0, 2 * Math.PI);
  ctx.stroke();
}

function createPlane(normal, pointOn) {
  let d = normal.dot(pointOn);
  return new THREE.Plane(normal, d);
}

function createPlaneAtOrigin(mesh, normal) {
  let ptL = new THREE.Vector3().fromArray(mesh[LMRK.tragion_L]);
  let ptR = new THREE.Vector3().fromArray(mesh[LMRK.tragion_R]);
  let p = ptL.clone().lerp(ptR, 0.5);
  return createPlane(normal, p);
}

function headPose(mesh) {
  let M = headCsysMoving(mesh); // Moving csys
  let F = headCsysCanonical();  // Fixed csys
  let f1 = new THREE.Vector3();
  let f2 = new THREE.Vector3();
  let f3 = new THREE.Vector3();
  F.extractBasis(f1, f2, f3);
  let m1 = new THREE.Vector3();
  let m2 = new THREE.Vector3();
  let m3 = new THREE.Vector3();
  M.extractBasis(m1, m2, m3);
  // Rotation matrix
  let r1 = new THREE.Vector3().set(m1.dot(f1), m1.dot(f2), m1.dot(f3));
  let r2 = new THREE.Vector3().set(m2.dot(f1), m2.dot(f2), m2.dot(f3));
  let r3 = new THREE.Vector3().set(m3.dot(f1), m3.dot(f2), m3.dot(f3));
  let RM = new THREE.Matrix4().makeBasis(r1, r2, r3); 
  // Head planes - Used for intersections
  let headPlanes = {
    'frontal'   : createPlaneAtOrigin(mesh, m3),
    'median'    : createPlaneAtOrigin(mesh, m1),
    'transverse': createPlaneAtOrigin(mesh, m2)
  };
  // Euler angles
  let eulerAnglesRad = new THREE.Euler().setFromRotationMatrix (RM, 'XYZ').toArray();
  let eulerAngles = [];
  for (let i = 0; i < 3; i++) {
    let a = THREE.MathUtils.radToDeg(eulerAnglesRad[i]);
    // NOTE: M is already rotated 180 degrees about F, so correct angles 
    // to within -90 to +90 range
    if (a >  90) {a = a - 180; }
    if (a < -90) {a = a + 180; }
    eulerAngles.push(a);
  }
  // Update html
  updateHeadPoseValues(eulerAngles);
  // Estimate nose width
  calculateNoseWidth(mesh, headPlanes);
  // Estimate nose depth
  calculateNoseDepth(mesh, headPlanes);  
  // Return
  return eulerAngles;
}

function calculateNoseWidth(mesh, headPlanes, tol=2.0) {
  let plane = headPlanes.frontal;
  let ray_origin_z  = 1e4;
  let ray_direction = new THREE.Vector3(0,0,-1);
  // Get points
  let pnL = new THREE.Vector3().fromArray(mesh[LMRK.nose_L]);
  let pnR = new THREE.Vector3().fromArray(mesh[LMRK.nose_R]);
  // Ray - nose_L
  let nL_origin = new THREE.Vector3(pnL.x, pnL.y, ray_origin_z);
  let nL_ray    = new THREE.Ray(nL_origin, ray_direction);
  // Ray - nose_R
  let nR_origin = new THREE.Vector3(pnR.x, pnR.y, ray_origin_z);
  let nR_ray    = new THREE.Ray(nR_origin, ray_direction);
  // Get intersection points
  let noseWidth = null;
  let is_perpendicular = THREE.MathUtils.radToDeg(Math.abs(plane.normal.dot(ray_direction))) < tol;
  if (!is_perpendicular) {
    if ((nL_ray.intersectsPlane(plane)) &&
        (nR_ray.intersectsPlane(plane))) {
      // nose_L
      let nL_intersect = new THREE.Vector3();
      nL_ray.intersectPlane(plane, nL_intersect);
      // nose_R
      let nR_intersect = new THREE.Vector3();
      nR_ray.intersectPlane(plane, nR_intersect); 
      // Get 3D distance between intersection points 
      noseWidth = nL_intersect.distanceTo(nR_intersect);
      // Check estimated value is within physical range
      //if (noseWidth < noseWidthRange.min) { noseWidth = noseWidthRange.min; }
      //if (noseWidth > noseWidthRange.max) { noseWidth = noseWidthRange.max; }
      // Print to console
      console.log("Nose width = " + noseWidth.toFixed(1));
    }
  }
  return noseWidth;
}

function calculateNoseDepth(mesh, headPlanes, tol=2.0) {
  // NOTE: Use either the median or transverse planes for noseDepth
  let plane = headPlanes.median;
  let ray_direction = new THREE.Vector3(0,0,-1);
  // Get points
  let pnL = new THREE.Vector3().fromArray(mesh[LMRK.nose_L]);
  let pnR = new THREE.Vector3().fromArray(mesh[LMRK.nose_R]);
  let pnT = new THREE.Vector3().fromArray(mesh[LMRK.nosetip]);
  let pnLRavg = pnL.clone().lerp(pnR, 0.5);
  // Ray - nose tip
  let nT_origin = new THREE.Vector3(pnT.x, pnT.y, 1e6);
  let nT_ray    = new THREE.Ray(nT_origin, ray_direction);
  // Ray - nose_LRavg
  let nLRavg_origin = new THREE.Vector3(pnLRavg.x, pnLRavg.y, 1e6);
  let nLRavg_ray    = new THREE.Ray(nLRavg_origin, ray_direction);  
  // Get intersection points
  let noseDepth = null;
  let is_perpendicular = THREE.MathUtils.radToDeg(Math.abs(plane.normal.dot(ray_direction))) < tol;
  if (!is_perpendicular) {
    if ((nT_ray.intersectsPlane(plane)) &&
        (nLRavg_ray.intersectsPlane(plane))) {
      // nose tip
      let nT_intersect = new THREE.Vector3();
      nT_ray.intersectPlane(plane, nT_intersect);
      // nose_LRavg
      let nLRavg_intersect = new THREE.Vector3();
      nLRavg_ray.intersectPlane(plane, nLRavg_intersect); 
      // Get 3D distance between intersection points 
      noseDepth = nT_intersect.distanceTo(nLRavg_intersect);
      // Check estimated value is within physical range
      //if (noseDepth < noseDepthRange.min) { noseDepth = noseDepthRange.min; }
      //if (noseDepth > noseDepthRange.max) { noseDepth = noseDepthRange.max; }
      // Print to console
      console.log("Nose depth = " + noseDepth.toFixed(1));
    }
  }
  return noseDepth;
}

function updateHeadPoseValues(eulerAngles) {
  // Un-pack euler angles
  let rotX = eulerAngles[0]; // Up-down (Down is +ve)
  let rotY = eulerAngles[1]; // Left-right (Left is +ve)
  let rotZ = eulerAngles[2]; // Rotate left-right (Right is +ve)
  // Up-down
  let facing_updown = rotX >= 0.0 ? 'DOWNWARDS' : 'UPWARDS';
  let angle_updown  = Math.abs(rotX);
  // Left-right
  let facing_leftright = rotY >= 0.0 ? 'LEFT' : 'RIGHT';
  let angle_leftright  = Math.abs(rotY);
  // Rotate_leftright
  let facing_rotate = rotZ >= 0.0 ? 'RIGHT' : 'LEFT';
  let angle_rotate  = Math.abs(rotZ);  
  // Head pose string
  let headPosition = "Head position: Turned ";
  headPosition += angle_leftright.toFixed(1) + " deg to the " + facing_leftright;
  headPosition += ", " + angle_updown.toFixed(1) + " deg " + facing_updown;
  headPosition += ", and rotated " + angle_rotate.toFixed(1) + " deg to the " + facing_rotate;
  document.getElementById('head-pose').innerHTML = headPosition;
}

function estimateNoseDepth(mesh, irisScaleFactor) {
  // Landmark coordinates
  let p_nt = mesh[LMRK.nosetip];
  let p_nl = tf.tensor1d(mesh[LMRK.nose_L], 'float32');
  let p_nr = tf.tensor1d(mesh[LMRK.nose_R], 'float32');
  let p_nm = p_nl.stack(p_nr).mean(0).arraySync();
  // Get xy distance between points. Apply irisScaleFactor to remove scaling effect
  // when user moves closer / further away from the camera
  let xy = distance(p_nt, p_nm) * irisScaleFactor;
  //console.log(xy);
  return xy;
}

function displayLandmarks(prediction) {

  // Function to display landmarks on the video frame
  
  // Get scaled mesh (x,y coords correspond to image pixel coords)
  const scaledMesh = prediction.scaledMesh;
  
  // Get landmark points - Only get x,y coordinates. Ignore z.
  let p_nose_L = new THREE.Vector3(scaledMesh[LMRK.nose_L].slice(0,2));
  let p_nose_R = new THREE.Vector3(scaledMesh[LMRK.nose_R].slice(0,2));
  let p_nosetip = new THREE.Vector3(scaledMesh[LMRK.nosetip].slice(0,2));
  let p_sellion = new THREE.Vector3(scaledMesh[LMRK.sellion].slice(0,2));
  let p_supramenton = new THREE.Vector3(scaledMesh[LMRK.supramenton].slice(0,2));
  let p_tragion_L = new THREE.Vector3(scaledMesh[LMRK.tragion_L].slice(0,2));
  let p_tragion_R = new THREE.Vector3(scaledMesh[LMRK.tragion_R].slice(0,2));

  // Plot landmarks
  plotLandmark(p_nose_L, WHITE);
  plotLandmark(p_nose_R, WHITE);
  plotLandmark(p_nosetip, WHITE); 
  plotLandmark(p_sellion, BLUE);
  plotLandmark(p_supramenton, BLUE);
  plotLandmark(p_tragion_L, RED);
  plotLandmark(p_tragion_R, RED); 

  // Get face measurements
  // Face height
  const faceHeightScaled = p_sellion.distanceTo(p_supramenton);
  // Face width (not needed for mask sizing, but maybe useful for conduit sizing)
  const faceWidthScaled = p_tragion_L.distanceTo(p_tragion_R);  
  // Nose width
  const noseWidthScaled = p_nose_L.distanceTo(p_nose_R);
  // Nose depth
  const noseDepthScaled = 0.5 * (p_nosetip.distanceTo(p_nose_L) + p_nosetip.distanceTo(p_nose_R));

  // Get iris
  let has_iris = scaledMesh.length > NUM_KEYPOINTS;
  if (has_iris) {

  }
}

function getLandmarkMeasurements(prediction) {

  /*
  //console.log(prediction);

  const mesh = prediction.mesh;
  const scaledMesh = prediction.scaledMesh;
  let x, y;

  // Face height
  const faceHeight = distance(
    mesh[LMRK.sellion],
    mesh[LMRK.supramenton]);

  const faceHeightScaled = distance(
    scaledMesh[LMRK.sellion],
    scaledMesh[LMRK.supramenton]);

  ctx.fillStyle = BLUE;
  ctx.strokeStyle = BLUE;
  ctx.lineWidth = 1;
  x = scaledMesh[LMRK.sellion][0];
  y = scaledMesh[LMRK.sellion][1];
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 2 * Math.PI);
  ctx.fill();
  x = scaledMesh[LMRK.supramenton][0];
  y = scaledMesh[LMRK.supramenton][1];
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 2 * Math.PI);
  ctx.fill();

  // Face width
  const faceWidth = distance(
    mesh[LMRK.tragion_L],
    mesh[LMRK.tragion_R]);

  const faceWidthScaled = distance(
    scaledMesh[LMRK.tragion_L],
    scaledMesh[LMRK.tragion_R]);

  ctx.fillStyle = RED;
  ctx.strokeStyle = RED;
  ctx.lineWidth = 1;
  x = scaledMesh[LMRK.tragion_L][0];
  y = scaledMesh[LMRK.tragion_L][1];
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 2 * Math.PI);
  ctx.fill();
  x = scaledMesh[LMRK.tragion_R][0];
  y = scaledMesh[LMRK.tragion_R][1];
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 2 * Math.PI);
  ctx.fill();

  // Nose width
  const noseWidth = distance(
    mesh[LMRK.nose_L],
    mesh[LMRK.nose_R]);

  const noseWidthScaled = distance(
    scaledMesh[LMRK.nose_L],
    scaledMesh[LMRK.nose_R]);

  ctx.fillStyle = WHITE;
  ctx.strokeStyle = WHITE;
  ctx.lineWidth = 1;
  x = scaledMesh[LMRK.nose_L][0];
  y = scaledMesh[LMRK.nose_L][1];
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 2 * Math.PI);
  ctx.fill();
  x = scaledMesh[LMRK.nose_R][0];
  y = scaledMesh[LMRK.nose_R][1];
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 2 * Math.PI);
  ctx.fill();

  // Nose depth
  const noseDepthL = distance(
    mesh[LMRK.nose_L],
    mesh[LMRK.nosetip]);
  const noseDepthR = distance(
    mesh[LMRK.nose_R],
    mesh[LMRK.nosetip]);
  const noseDepth = 0.5 * (noseDepthL + noseDepthR);

  const noseDepthScaledL = distance(
    scaledMesh[LMRK.nose_L],
    scaledMesh[LMRK.nosetip]);
  const noseDepthScaledR = distance(
    scaledMesh[LMRK.nose_R],
    scaledMesh[LMRK.nosetip]);
  const noseDepthScaled = 0.5 * (noseDepthScaledL + noseDepthScaledR);

  ctx.fillStyle = WHITE;
  ctx.strokeStyle = WHITE;
  ctx.lineWidth = 1;
  x = scaledMesh[LMRK.nosetip][0];
  y = scaledMesh[LMRK.nosetip][1];
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 2 * Math.PI);
  ctx.fill();
  */

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

async function run() {

  stats.begin();
  const predictions = await model.estimateFaces(
    video, false /* returnTensors */, false /* flipHorizontal */,
    state.predictIrises);
  ctx.drawImage(
    video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);  

  if (predictions.length > 0) {
    
    // Display mesh and irises
    predictions.forEach(prediction => {
      renderPrediction(prediction);
      displayLandmarks(prediction);
      headPose(prediction.scaledMesh);
    });

    scatterPlot(predictions);
  }
  stats.end();
  rafID = requestAnimationFrame(run);
};

function renderPrediction(prediction) {

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

  // Show irises
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
  //headPose(prediction.scaledMesh);
  //let hp = headPose(prediction.scaledMesh);
  //let poseLR = hp[0];
  //let poseUD = hp[1];

  /*
  // Plots landmarks and update landmark measurements
  let irisScaleFactor = getLandmarkMeasurements(prediction);
  if (irisScaleFactor != null) {

    // Estimate nose depth
    let angleLR = Math.abs(poseLR);
    let angleUD = Math.abs(poseUD);
    if ((angleLR >= 10.0) && (angleLR <= 30.0) && (angleUD <= 10.0)) {

      let xyDistance = estimateNoseDepth(prediction.scaledMesh, irisScaleFactor);

      if (noseDepthEstimate.values.length < 50) {
        let nd = xyDistance / Math.sin(THREE.MathUtils.degToRad(angleLR));
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
  }*/
}

function scatterPlot(predictions) {

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
  run();

  if (renderPointcloud) {
    document.querySelector('#scatter-gl-container').style =
      `width: ${VIDEO_SIZE}px; height: ${VIDEO_SIZE}px;`;

    scatterGL = new ScatterGL(
      document.querySelector('#scatter-gl-container'),
      { 'rotateOnStart': false, 'selectEnabled': false });
  }
};

main();
