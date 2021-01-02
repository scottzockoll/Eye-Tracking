import {load, SupportedPackages} from '@tensorflow-models/face-landmarks-detection';
import p5 from 'p5';
import * as tf from '@tensorflow/tfjs'
import {buildModel} from './neuralNetwork'

let model, videoWidth, videoHeight, video, canvas;
let neuralNetwork = buildModel()

const VIDEO_SIZE = 500;
const state = {
    backend: 'webgl',
    maxFaces: 1,
    triangulateMesh: true,
    predictIrises: true,
};

function distance(a, b) {
    return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

async function getPredictions() {
    return model.estimateFaces({
        input: document.querySelector('video'),
    });
}

async function getNeuralNetworkPredictions(x) {
    x = tf.tensor2d(x, [1, 30])
    x.print()
    return neuralNetwork.predict(x)
}

async function setupCamera() {
    video = document.getElementById('video');

    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            width: VIDEO_SIZE,
            height: VIDEO_SIZE,
        },
    })
    video.srcObject = stream

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function setUp() {
    await tf.setBackend(state.backend)

    await setupCamera();
    video.play();
    videoWidth = video.videoWidth;
    videoHeight = video.videoHeight;
    video.width = videoWidth;
    video.height = videoHeight;
    console.log('camera loaded')

    model = await load(
        SupportedPackages.mediapipeFacemesh,
        {maxFaces: state.maxFaces});

    getPredictions().then(() => {
        console.log('Model loaded')
    })

    return model

}

let training = false
let predicting = false
let guiding = false
let guideX = 0
let guideY = 0
let guideSign = 1
let guideWidth = 50
let X = []
let y = []
const EPOCHS = 50;

function train(model, X, y) {
    console.log('training neural network')
    model.fit(tf.tensor2d(X, [X.length, 30]), tf.tensor2d(y, [y.length, 2]), {
        epochs: EPOCHS
    }).then(() => {
        console.log('neural network trained')
    });
    return model
}

async function main() {
    console.log('loading camera and model...')
    model = await setUp()

    const sketch = (s) => {

        s.setup = () => {
            let canvas = s.createCanvas(s.windowWidth, s.windowHeight);
            canvas.parent('p5jsdiv');
        };

        if (running) {
            s.draw = () => {
                if (guiding) {
                    s.background(255)
                    if (((guideSign > 0) && (guideX >= s.windowWidth-guideWidth)) || ((guideSign < 0) && (guideX <= 0))) {
                        guideY = guideY + 50
                        guideSign = guideSign > 0 ? -1 : 1
                    } else {
                        guideX = guideX + (20 * guideSign)
                    }
                    s.translate(guideX, guideY);
                    s.fill(0);
                    s.rect(0, 0, guideWidth, guideWidth);
                }
                getPredictions().then((prediction) => {

                    // start paste
                    let annotations = prediction[0]['annotations']
                    let leftPoints = annotations['leftEyeIris']
                    let rightPoints = annotations['rightEyeIris']
                    let leftCenter = leftPoints[0]
                    let rightCenter = rightPoints[0]

                    let xCompOfLeftCenterDot = VIDEO_SIZE - leftCenter[0]
                    let yCompOfLeftCenterDot = leftCenter[1]
                    let xOfLockedLeftCenterDot = xCompOfLeftCenterDot - (xCompOfLeftCenterDot - 100)
                    let yOfLockedLeftCenterDot = yCompOfLeftCenterDot - (yCompOfLeftCenterDot - 100)

                    let xCompOfRightCenterDot = VIDEO_SIZE - rightCenter[0]
                    let yCompOfRightCenterDot = rightCenter[1]
                    let xOfLockedRightCenterDot = xCompOfRightCenterDot - (xCompOfRightCenterDot - 150)
                    let yOfLockedRightCenterDot = yCompOfRightCenterDot - (yCompOfRightCenterDot - 100)

                    // end paste

                    if (guiding) {
                        let dataX = []
                        leftPoints.forEach((point, index) => {
                            // plot unlocked dot

                            // plot locked dot
                            let xCompOfDot = VIDEO_SIZE - point[0]
                            let yCompOfDot = point[1]
                            let normalizedX = (xCompOfDot - (xCompOfDot - xOfLockedLeftCenterDot)) + (xCompOfDot - xCompOfLeftCenterDot)
                            let normalizedY = (yCompOfDot - (yCompOfDot - yOfLockedLeftCenterDot)) + (yCompOfDot - yCompOfLeftCenterDot)
                            dataX.push(normalizedX)
                            dataX.push(normalizedY)
                            dataX.push(point[2])

                        })

                        rightPoints.forEach((point, index) => {

                            // plot locked dot
                            let xCompOfDot = VIDEO_SIZE - point[0]
                            let yCompOfDot = point[1]
                            let normalizedX = (xCompOfDot - (xCompOfDot - xOfLockedRightCenterDot)) + (xCompOfDot - xCompOfRightCenterDot)
                            let normalizedY = (yCompOfDot - (yCompOfDot - yOfLockedRightCenterDot)) + (yCompOfDot - yCompOfRightCenterDot)
                            dataX.push(normalizedX)
                            dataX.push(normalizedY)
                            dataX.push(point[2])
                        })
                        X.push(dataX)
                        y.push([guideX, guideY])
                    }

                    if (training) {
                        let dataX = []
                        leftPoints.forEach((point, index) => {
                            // plot unlocked dot

                            // plot locked dot
                            let xCompOfDot = VIDEO_SIZE - point[0]
                            let yCompOfDot = point[1]
                            let normalizedX = (xCompOfDot - (xCompOfDot - xOfLockedLeftCenterDot)) + (xCompOfDot - xCompOfLeftCenterDot)
                            let normalizedY = (yCompOfDot - (yCompOfDot - yOfLockedLeftCenterDot)) + (yCompOfDot - yCompOfLeftCenterDot)
                            dataX.push(normalizedX)
                            dataX.push(normalizedY)
                            dataX.push(point[2])

                        })

                        rightPoints.forEach((point, index) => {

                            // plot locked dot
                            let xCompOfDot = VIDEO_SIZE - point[0]
                            let yCompOfDot = point[1]
                            let normalizedX = (xCompOfDot - (xCompOfDot - xOfLockedRightCenterDot)) + (xCompOfDot - xCompOfRightCenterDot)
                            let normalizedY = (yCompOfDot - (yCompOfDot - yOfLockedRightCenterDot)) + (yCompOfDot - yCompOfRightCenterDot)
                            dataX.push(normalizedX)
                            dataX.push(normalizedY)
                            dataX.push(point[2])
                        })
                        X.push(dataX)
                        y.push([s.mouseX, s.mouseY])
                    }
                    if (predicting) {
                        s.background('rgba(255,255,255,0.15)');
                        let dataX = []
                        leftPoints.forEach((point, index) => {
                            // plot unlocked dot

                            // plot locked dot
                            let xCompOfDot = VIDEO_SIZE - point[0]
                            let yCompOfDot = point[1]
                            let normalizedX = (xCompOfDot - (xCompOfDot - xOfLockedLeftCenterDot)) + (xCompOfDot - xCompOfLeftCenterDot)
                            let normalizedY = (yCompOfDot - (yCompOfDot - yOfLockedLeftCenterDot)) + (yCompOfDot - yCompOfLeftCenterDot)
                            dataX.push(normalizedX)
                            dataX.push(normalizedY)
                            dataX.push(point[2])

                        })

                        rightPoints.forEach((point, index) => {

                            // plot locked dot
                            let xCompOfDot = VIDEO_SIZE - point[0]
                            let yCompOfDot = point[1]
                            let normalizedX = (xCompOfDot - (xCompOfDot - xOfLockedRightCenterDot)) + (xCompOfDot - xCompOfRightCenterDot)
                            let normalizedY = (yCompOfDot - (yCompOfDot - yOfLockedRightCenterDot)) + (yCompOfDot - yCompOfRightCenterDot)
                            dataX.push(normalizedX)
                            dataX.push(normalizedY)
                            dataX.push(point[2])
                        })
                        getNeuralNetworkPredictions(dataX).then((x_y) => {
                            x_y.array().then((array) => {
                                console.log(array)
                                let x_prediction = array[0][0]
                                let y_prediction = array[0][1]
                                s.drawPoint(x_prediction, y_prediction, 'green')
                            })
                        })
                    }
                })
            }
        }


        s.mouseClicked = (e) => {
            console.log(e.screenX, e.screenY)
        }

        s.outlineIris = (points) => {
            const diameterX = distance(
                points[3],
                points[1]
            );
            const diameterY = distance(
                points[2],
                points[4]
            );
            s.noFill()
            s.ellipse(VIDEO_SIZE - points[0][0], points[0][1], diameterX / 2, diameterY / 2)
        }

        s.drawPoint = (x, y, color) => {
            // let new_z = -(z / 1.5) * 255
            if (color === 'green') {
                s.fill(0, 250, 0)
            } else if (color === 'blue') {
                s.fill(0, 0, 250)
            } else if (color === 'red') {
                s.fill(250, 0, 0)
            } else {
                console.log(`${color} is not a supported color. Using green.`)
                s.fill(0, 250, 0)
            }

            s.circle(x, y, 10)
        }

        s.keyPressed = (e) => {
            if (e.key === 't') {
                if (training) {
                    training = false
                    console.log('training stopped')
                } else {
                    training = true
                    console.log('training started')
                }
            } else if (e.key === 'd') {
                console.log(X)
                console.log(y)
            } else if (e.key === 'n') {
                neuralNetwork = train(neuralNetwork, X, y)
            } else if (e.key === 'p') {
                if (predicting) {
                    predicting = false
                    console.log('predicting stopped')
                } else {
                    predicting = true
                    console.log('prediction started')

                }
            } else if (e.key === 'g') {
                if (guiding) {
                    guiding = false
                    console.log('guiding stopped')
                } else {
                    guiding = true
                    console.log('guiding started')
                }
            }
        }

    };

    new p5(sketch);

}

const running = true
main()





