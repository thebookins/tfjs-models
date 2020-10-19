const faceLandmarksDetection = require('@tensorflow-models/face-landmarks-detection');
const fs = require('fs');
const pixels = require('image-pixels');

const three = require('three');

// Landmark list
const LMRK = {
  infraorb_L: 330,
  infraorb_R: 101,
  nose_alar_L: 278,             // For nose width
  nose_alar_R: 48,              // For nose width
  nose_alarfacialgroove_L: 358, // For nose depth
  nose_alarfacialgroove_R: 129, // For nose depth
  nose_tip: 4,                  // For nose depth
  sellion: 168,
  supramenton: 200,
  tragion_L: 454,
  tragion_R: 234,
  iris_top_L: 470,
  iris_bottom_L: 472
};


// If you are using the WebGL backend:
//require('@tensorflow/tfjs-backend-wasm');

tf = require('@tensorflow/tfjs-core');
require('@tensorflow/tfjs-backend-webgl');
require('@tensorflow/tfjs-backend-cpu');
//tf = require('@tensorflow/tfjs-node')


async function main() {
  await tf.setBackend('cpu');

  // Load the MediaPipe Facemesh package.
  //const model = await faceLandmarksDetection.load(config = {
  //  inputWidth: 640,
  //  inputHeight: 480
  //});

  const model = await faceLandmarksDetection.load();

  const img = await pixels('image.jpg');

  const predictions = await model.estimateFaces({
    input: img
  });

  const prediction = predictions[0]
  const distanceXY = (a, b) => Math.sqrt(Math.pow((a.x - b.x), 2) + Math.pow((a.y - b.y), 2));

  const faceHeight = distanceXY(
    new three.Vector3().fromArray(prediction.mesh[LMRK.sellion]),
    new three.Vector3().fromArray(prediction.mesh[LMRK.supramenton])
  );

  const noseWidth = distanceXY(
    new three.Vector3().fromArray(prediction.mesh[LMRK.nose_alar_L]),
    new three.Vector3().fromArray(prediction.mesh[LMRK.nose_alar_R])
  );

  const irisDiam = distanceXY(
    new three.Vector3().fromArray(prediction.mesh[LMRK.iris_top_L]),
    new three.Vector3().fromArray(prediction.mesh[LMRK.iris_bottom_L])
  );

  console.log(`noseWidth = ${noseWidth}`)
  console.log(`faceHeight = ${faceHeight}`)
  console.log(`irisDiam = ${irisDiam}`)

  const scaleFactor = 11.7 / irisDiam;

  console.log(`noseWidthNorm = ${noseWidth * scaleFactor}`)
  console.log(`faceHeightNorm = ${faceHeight * scaleFactor}`)

  fs.writeFileSync('test.json', JSON.stringify(predictions));
}

main();
