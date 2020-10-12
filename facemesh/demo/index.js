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
import { MESH_ANNOTATIONS } from '../dist/keypoints';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;
const GREEN = '#32EEDB';
const RED = "#FF2C35";
const BLUE = "#0000FF";
const WHITE = "#FFFFFF";

// Iris scaling
const scaleFactor = 1.0;
const IRIS_DIAMETER_AVGERAGE = 11.7 * scaleFactor;

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

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
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
      requestAnimationFrame(run);
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

function average(data) {
  // Average of all values in the data array
  return data.reduce((a,b) => (a+b)) / data.length;
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

function distanceXY(a, b) {
  // Computes the distance in the X-Y plane of points a and b
  // a, b: Either Vector2 or Vector3
  return Math.sqrt(Math.pow((a.x-b.x), 2) + Math.pow((a.y-b.y), 2));
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

function plotIris(center, diameter, colour=RED, lineWidth=1) {
  let radius = diameter / 2;
  ctx.strokeStyle = colour;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.ellipse(center.x, center.y, radius, radius, 0, 0, 2 * Math.PI);
  ctx.stroke();
}

function plotLandmark(lmrk, colour=BLUE, radius=3) {
  ctx.fillStyle = colour;
  ctx.strokeStyle = colour;
  ctx.beginPath();
  ctx.arc(lmrk.x, lmrk.y, radius, 0, 2 * Math.PI);
  ctx.fill();
}

function getHeadPose(mesh) {
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
  return [eulerAngles, headPlanes]
}

function calculateNoseWidth(mesh, plane, tol=2.0) {
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
    }
  }
  return noseWidth;
}

function calculateNoseDepth(mesh, plane, tol=2.0) {
  // NOTE: Use either the median or transverse planes for noseDepth
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

function getHeadMeasures(mesh, plotLandmarks=true) {

  // Function to display landmarks on the video frame
  
  // Get landmark points
  let p_infraorb_L = new THREE.Vector3().fromArray(mesh[LMRK.infraorb_L]);
  let p_infraorb_R = new THREE.Vector3().fromArray(mesh[LMRK.infraorb_R]);  
  let p_nose_L = new THREE.Vector3().fromArray(mesh[LMRK.nose_L]);
  let p_nose_R = new THREE.Vector3().fromArray(mesh[LMRK.nose_R]);
  let p_nosetip = new THREE.Vector3().fromArray(mesh[LMRK.nosetip]);
  let p_sellion = new THREE.Vector3().fromArray(mesh[LMRK.sellion]);
  let p_supramenton = new THREE.Vector3().fromArray(mesh[LMRK.supramenton]);
  let p_tragion_L = new THREE.Vector3().fromArray(mesh[LMRK.tragion_L]);
  let p_tragion_R = new THREE.Vector3().fromArray(mesh[LMRK.tragion_R]);

  // Plot landmarks
  if (plotLandmarks) {
    plotLandmark(p_infraorb_L, BLUE);
    plotLandmark(p_infraorb_R, BLUE);
    plotLandmark(p_nose_L, WHITE);
    plotLandmark(p_nose_R, WHITE);
    plotLandmark(p_nosetip, WHITE);
    plotLandmark(p_sellion, BLUE);
    plotLandmark(p_supramenton, BLUE);
    plotLandmark(p_tragion_L, RED);
    plotLandmark(p_tragion_R, RED);
  }

  // Get face measurements
  // Face height
  const faceHeight = distanceXY(p_sellion, p_supramenton);
  // Face width (not needed for mask sizing, but maybe useful for conduit sizing)
  const faceWidth = distanceXY(p_tragion_L, p_tragion_R);  
  // Nose depth
  const noseDepth = average([distanceXY(p_nosetip, p_nose_L), distanceXY(p_nosetip, p_nose_R)]);
  // Nose width
  const noseWidth = distanceXY(p_nose_L, p_nose_R);
  // Landmarks measurements
  let headMeasures = {
    'faceHeight' : faceHeight,
    'faceWidth'  : faceWidth,
    'noseDepth'  : noseDepth,
    'noseWidth'  : noseWidth
  };
  return headMeasures
}

function getIrisMeasures(scaledMesh) {

  // Get iris measures
  let irisDiameters = null;
  let has_iris = scaledMesh.length > NUM_KEYPOINTS;
  if (has_iris) {
    // Get coordinates of iris keypoints
    let leftIris  = MESH_ANNOTATIONS['leftEyeIris'].map(i=> new THREE.Vector3().fromArray(scaledMesh[i]));
    let rightIris = MESH_ANNOTATIONS['rightEyeIris'].map(i=> new THREE.Vector3().fromArray(scaledMesh[i]));
    // Plot left and right iris keypoints
    leftIris.map(lmrk=> plotLandmark(lmrk, RED, 2));
    rightIris.map(lmrk=> plotLandmark(lmrk, RED, 2));
    // Get iris diameter
    // Left iris
    let leftRadius = [];
    for (let i=1; i<MESH_ANNOTATIONS['leftEyeIris'].length; i++) { 
      leftRadius.push(leftIris[i].distanceTo(leftIris[0])); }     // Should be distanceXY 
    let leftIrisDiameter = average(leftRadius) * 2;
    // Right iris
    let rightRadius = [];
    for (let i=1; i<MESH_ANNOTATIONS['rightEyeIris'].length; i++) { 
      rightRadius.push(rightIris[i].distanceTo(rightIris[0])); }  // Should be distanceXY 
    let rightIrisDiameter = average(rightRadius) * 2;
    // Min, max and average iris diameters 
    let avgIrisDiameter = average([leftIrisDiameter, rightIrisDiameter]);
    let minIrisDiameter = leftIrisDiameter < rightIrisDiameter ? leftIrisDiameter : rightIrisDiameter;
    let maxIrisDiameter = leftIrisDiameter > rightIrisDiameter ? leftIrisDiameter : rightIrisDiameter;
    // Iris scale factor
    let irisScaleFactor = IRIS_DIAMETER_AVGERAGE / maxIrisDiameter;
    // Store iris diameters
    irisDiameters = {
      'left'  : leftIrisDiameter,
      'right' : rightIrisDiameter,
      'min'   : minIrisDiameter,
      'max'   : maxIrisDiameter,
      'avg'   : avgIrisDiameter,
      'scale' : irisScaleFactor
    };
    // Show iris 
    plotIris(leftIris[0], leftIrisDiameter, RED);
    plotIris(rightIris[0], rightIrisDiameter, RED);
  }
  return irisDiameters
}

function updateHeadMeasureValues(headMeasures, irisDiameters) {

  // Head measurements
  const face_height = document.getElementById('measure-face-height');
  const face_width  = document.getElementById('measure-face-width');
  const nose_width  = document.getElementById('measure-nose-width');
  const nose_depth  = document.getElementById('measure-nose-depth');
  face_height.innerHTML = "Sellion-supramenton = " + headMeasures.faceHeight.toFixed(1) + " mm";
  face_width.innerHTML  = "Face width = " + headMeasures.faceWidth.toFixed(1) + " mm";
  nose_width.innerHTML  = "Nose width = " + headMeasures.noseWidth.toFixed(1) + " mm";
  nose_depth.innerHTML  = "Nose depth = " + headMeasures.noseDepth.toFixed(1) + " mm";
  
  // Iris diameters
  const iris_diam_L = document.getElementById('measure-scaled-iris-diameter-L');
  const iris_diam_R = document.getElementById('measure-scaled-iris-diameter-R');
  if (irisDiameters) {
    iris_diam_L.innerHTML = "Scaled iris diameter L = " + irisDiameters.left.toFixed(1);
    iris_diam_R.innerHTML = "Scaled iris diameter R = " + irisDiameters.right.toFixed(1);  
  } else {
    iris_diam_L.innerHTML = "";
    iris_diam_R.innerHTML = "";
  }
}

function renderFacemesh(keypoints) {

  // Display the face mesh (points or triangulated mesh) superimposed
  // over the face in the video feed

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
      ctx.arc(x, y, 1, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
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

function checkMeasuredValues(headMeasures) {
  
  // Ensure that the head measurements are within physically 
  // realistic ranges
  
  // Nose width
  if (headMeasures.noseWidth < noseWidthRange.min) {
    headMeasures.noseWidth = noseWidthRange.min; }
  if (headMeasures.noseWidth > noseWidthRange.max) {
    headMeasures.noseWidth = noseWidthRange.max; }
  
  // Nose depth
  if (headMeasures.noseDepth < noseDepthRange.min) {
    headMeasures.noseDepth = noseDepthRange.min; }
  if (headMeasures.noseDepth > noseDepthRange.max) {
    headMeasures.noseDepth = noseDepthRange.max; }
  
  return headMeasures;
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

      const mesh = prediction.mesh;
      const scaledMesh = prediction.scaledMesh;
      
      // Render the face mesh (points or triangulated mesh)
      renderFacemesh(scaledMesh);   

      // Get head orientation (could use mesh or scaledMesh)
      let headPose = getHeadPose(scaledMesh);
      let eulerAngles = headPose[0];
      let headPlanes  = headPose[1];
      // Update head pose values in html
      updateHeadPoseValues(eulerAngles);

      // Head measurements v1 - Raw data that comes from facemesh model
      let headMeasuresv1 = getHeadMeasures(mesh, false);

      // Head measurements v2 - scaledMesh scaled using iris diameter
      // Get face and head landmarks measurements
      let headMeasuresv2 = getHeadMeasures(scaledMesh);
      // Get iris diameter data
      let irisMeasures = getIrisMeasures(scaledMesh);
      // Scale head measures using iris diameter
      if (irisMeasures) {
        for (const [key, value] of Object.entries(headMeasuresv2)) {
          headMeasuresv2[key] = value * irisMeasures.scale;
      }}
      // Check measured values are physical
      headMeasuresv2 = checkMeasuredValues(headMeasuresv2);
      // Update html with head measures and iris measures
      updateHeadMeasureValues(headMeasuresv2, irisMeasures);      

      // Head measurements v3 - Same as v2, but with projections onto head
      // planes to improve the accuracy of nose depth  
      // Estimate nose width
      let noseWidth = calculateNoseWidth(scaledMesh, headPlanes.frontal);
      let noseWidth_transverse = calculateNoseWidth(scaledMesh, headPlanes.transverse);
      // Estimate nose depth
      let noseDepth = calculateNoseDepth(scaledMesh, headPlanes.median);
      let noseDepth_transverse = calculateNoseDepth(scaledMesh, headPlanes.transverse);
      // Scale measurements using iris diameter
      if (irisMeasures) {
        if (noseWidth) { noseWidth *= irisMeasures.scale; }
        if (noseDepth) { noseDepth *= irisMeasures.scale; }
        if (noseWidth_transverse) {noseDepth_transverse *= irisMeasures.scale; }
      }
      // Print
      if (noseWidth) { console.log("Nose width = " + noseWidth.toFixed(1) + " mm"); }
      if (noseDepth) { console.log("Nose depth = " + noseDepth.toFixed(1) + " mm"); }
      if (noseWidth_transverse) { console.log("Nose width transverse = " + noseWidth_transverse.toFixed(1) + " mm"); }
      // Calculate diag nose depth
      if (noseWidth && noseDepth) {
        let noseWidthHalf = noseWidth / 2.0;
        let noseDepthDiag = Math.sqrt(Math.pow(noseWidthHalf,2) + Math.pow(noseDepth,2));
        console.log("Nose depth diagonal = " + noseDepthDiag.toFixed(1) + " mm");
      } 
    });

    // Show the scatter plot
    scatterPlot(predictions);

  }
  stats.end();
  rafID = requestAnimationFrame(run);
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
