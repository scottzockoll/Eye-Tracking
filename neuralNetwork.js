import * as tf from '@tensorflow/tfjs'

console.log('neural network loaded')

// Solve for XOR
const LEARNING_RATE = 0.01;

export function buildModel() {
    // Define the model.
    const model = tf.sequential();
    // Set up the network tf.layers
    model.add(tf.layers.dense({units: 8, activation: 'relu', inputShape: [30], useBias: true}));
    model.add(tf.layers.dense({units: 16, activation: 'relu', useBias: true}));
    model.add(tf.layers.dense({units: 32, activation: 'relu', useBias: true}));
    model.add(tf.layers.dense({units: 16, activation: 'relu', useBias: true}));
    model.add(tf.layers.dense({units: 8, activation: 'relu', useBias: true}));
    model.add(tf.layers.dense({units: 2, useBias: true}));
    // Define the optimizer
    const optimizer = tf.train.adam(LEARNING_RATE);
    // Init the model
    model.compile({
        optimizer: optimizer,
        loss: tf.losses.meanSquaredError,
        metrics: ['accuracy'],

    });
    return model
}





