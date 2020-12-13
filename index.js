import {load, SupportedPackages} from '@tensorflow-models/face-landmarks-detection';
import {setBackend} from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
// import '@tensorflow/tfjs-backend-cpu';
import p5 from 'p5';

let model, videoWidth, videoHeight, video, canvas;

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
    await setBackend(state.backend);

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


async function main() {
    console.log('loading camera and model...')
    model = await setUp()

    const sketch = (s) => {

        s.setup = () => {
            let canvas = s.createCanvas(VIDEO_SIZE, VIDEO_SIZE);
            canvas.parent('main');

        };

        if (running) {
            s.draw = () => {
                s.background('rgba(255,255,255,0.15)');
                getPredictions().then((prediction) => {
                    let annotations = prediction[0]['annotations']
                    let leftPoints = annotations['leftEyeIris']
                    let rightPoints = annotations['rightEyeIris']


                    leftPoints.forEach((point, index) => {
                        s.drawPoint(VIDEO_SIZE - point[0], point[1], point[2], 'green')

                    })

                    rightPoints.forEach((point) => {
                        s.drawPoint(VIDEO_SIZE - point[0], point[1], point[2], 'red')
                    })
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

        s.drawPoint = (x, y, z, color) => {
            let new_z = -(z / 1.5) * 255
            if (color === 'green') {
                s.fill(0, 250, 0, new_z)
            } else if (color === 'blue') {
                s.fill(0, 0, 250, new_z)
            } else if (color === 'red') {
                s.fill(250, 0, 0, new_z)
            } else {
                console.log(`${color} is not a supported color. Using green.`)
                s.fill(0, 250, 0, new_z)
            }

            s.circle(x, y, 10)
        }

        s.keyPressed = (e) => {
            if (e.key === 'z') {
                console.log('z pressed')
            }
        }

    };

    new p5(sketch);

}

const running = true
main()





