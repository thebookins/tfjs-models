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
    getValues() {
        return this.data;
    }
    update(new_value) {
        if (this.data.length == this.max_length) {
        this.data.shift();
        }
        this.data.push(new_value);
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