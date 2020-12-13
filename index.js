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

async function setupCamera() {
    video = document.getElementById('video');

    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            width: VIDEO_SIZE,
            height: VIDEO_SIZE,
        },
    }).then((stream) => {
        console.log('video loaded')
        video.srcObject = stream
    });

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function main() {
    await setBackend(state.backend);

    await setupCamera();
    video.play();
    videoWidth = video.videoWidth;
    videoHeight = video.videoHeight;
    video.width = videoWidth;
    video.height = videoHeight;

    model = await load(
        SupportedPackages.mediapipeFacemesh,
        {maxFaces: state.maxFaces});

    console.log('model loaded')

    return model;
}

model = main();

async function getPredictions() {
    return model.estimateFaces({
        input: document.querySelector('video'),
    });
}

const sketch = (s) => {

    s.setup = () => {
        let canvas = s.createCanvas(VIDEO_SIZE, VIDEO_SIZE);
        canvas.parent('main');

    };

    s.draw = () => {
        s.background('rgba(255,255,255,0.15)');
        getPredictions().then((prediction) => {
            let annotations = prediction[0]['annotations']
            let left = annotations['leftEyeIris']
            let right = annotations['rightEyeIris']
            left.forEach((point) => {
                s.drawPoint(VIDEO_SIZE - point[0], point[1], point[2], 'green')
            })
            right.forEach((point) => {
                s.drawPoint(VIDEO_SIZE - point[0], point[1], point[2], 'red')
            })
        })
    }

    s.mouseClicked = (e) => {
        console.log(e.screenX, e.screenY)
    }

    s.drawPoint = (x, y, z, color) => {
        let new_z = -(z / 1.5) * 255
        if (color === 'green') {
            s.fill(0, 250, 0, new_z)
        } else {
            s.fill(250, 0, 0, new_z)
        }

        s.circle(x, y, 10)
    }

    s.keyPressed = (e) => {
        if (e.key === 'z') {
            console.log('z pressed')
        }
    }

};

const sketchInstance = new p5(sketch);


