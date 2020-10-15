const express = require('express');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

const app = express();
app.use(express.static('./dist'));
app.get('/*', (req, res) =>
  res.sendFile('index.html', { root: 'dist/' }),
);

const server = require('http').createServer(app);
const io = require('socket.io')(server);
io.on('connection', (client) => {
  const headMeasures = [];
  const sessionID = uuidv4();
  client.on('head measures', (data) => {
    // the head measures are available here, to do with what you will
    // in this example they are added to an array
    // and saved on disconnect
    // (prob not the best way as the array can grow in memory indefitely)
    headMeasures.push(data);
  });
  client.on('disconnect', () => {
    console.log('disconnect');
    fs.writeFileSync(`./${sessionID}.json`, JSON.stringify(headMeasures));
  });
});
server.listen(process.env.PORT || 8080);
