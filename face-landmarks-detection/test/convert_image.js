const fs = require('fs');
tf = require('@tensorflow/tfjs-node')

async function main() {
  //await tf.setBackend('cpu');

  const serializedString = fs.readFileSync('leftEyeCrop.json')

  const dataArray = JSON.parse(serializedString);
  console.log(dataArray[0][0][0]);

  const imageData = dataArray[0]

  //var x = zeros([192, 192, 3])
  for (var i = 0; i < imageData.length; i++) {
    row = imageData[i]
    for (var j = 0; j < row.length; j++) {
      pixel = row[j]
      for (var k = 0; k < pixel.length; k++) {
        pixel[k] *= 256
        //console.log(dataArray[0][i][j][k] * 255)
        //x.set(i, j, k, dataArray[0][i][j][k] * 255)
      }
    }
  }

  ////Save to a file
  //file = fs.createWriteStream('face.png');
  //savePixels(x, "png").pipe(file)

  //savePixels(dataArray[0], "png").pipe(console.log);

  //var options = { colorType: 2, width: 192, height: 192, bitDepth: 8, inputColorType: 2 };
  //var buffer = PNG.sync.write(dataArray[0], options);
  //fs.writeFileSync('out.png', buffer);

  ////t = tf.cast(t, 'uint8')

  //console.log(t);

  t = tf.tensor(imageData)
  const png = await tf.node.encodePng(t, 0);


  //console.log(png);

  const data = Buffer.from(png);
  fs.writeFileSync('leftEyeCrop.png', data, 'binary');


  //const img = await pixels('image.jpg');
  //console.log("Image PixelData: ");
  //console.log(img);

  //const imageBuffer = fs.readFileSync('image.jpg');
  //tensor = tf.node.decodeImage(imageBuffer) // create a tensor for the image

  //const image = { width: 640, height: 480, data: imageBuffer }

  //console.log(imageBuffer.length)
}

main();
