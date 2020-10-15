// file system for writing json files
const fs = require('browserify-fs');

export class MovingAverage {

    constructor(max_length=50) {
        this.data = [];
        this.max_length = max_length; 
    }
    average() {
        return this.data.reduce((a,b) => (a+b)) / this.data.length;
    }
    clear() {
        this.data = [];
    }
    getLastValue() {
        return this.data[this.data.length-1];
    }
    getValues() {
        return this.data;
    }
    update(new_value) {
        if (this.data.length == this.max_length) {
        this.data.shift(); // Removes first value in array
        }
        this.data.push(new_value); // Adds new value to the end of array
    }
}

export function average(data) {
    // Average of all values in the data array
    return data.reduce((a,b) => (a+b)) / data.length;
}

export function distanceXY(a, b) {
    // Computes the distance in the X-Y plane of points a and b
    // a, b: Either Vector2 or Vector3
    return Math.sqrt(Math.pow((a.x-b.x), 2) + Math.pow((a.y-b.y), 2));
}

export class ScanMeasurements {

    constructor(filename='data.csv', numberOfFrames=50) {
        this.filename = filename;
        this.numFrames = numberOfFrames;
        this.currentFrame = 0;
        this.data = {
            'noseWidth'  : [],
            'noseDepth'  : [],
            'faceHeight' : []
        };
    }
    updateData(currentData) {
        // currentData must be an object with the same keys as
        // this.data, representing the values at the current frame
        for (const [key, value] of Object.entries(currentData)) {
            if (Object.keys(this.data).includes(key)) {
                this.data[key].push(value);
            }
        }
        this.currentFrame += 1;
        if (this.currentFrame == this.numFrames) {
            this.writeDataToFile();
        }
    }
    writeDataToFile() {
        let data = JSON.stringify(this.data);
        fs.writeFile(this.filename, data, (err) => {
            if (err) throw err;
            console.log('Data written to file');
        });
    }
}