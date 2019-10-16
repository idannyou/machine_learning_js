/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const tf = require('@tensorflow/tfjs-node');
const argparse = require('argparse');

const data = require('./data');
const model = require('./model');

async function run(epochs, batchSize, modelSavePath, modelLoadPath) {
  await data.loadData();

  if (modelLoadPath != null) {
    const loadedModel = await tf.loadLayersModel(`file://${modelLoadPath}`);
    loadedModel.compile({
      loss: 'categoricalCrossentropy',
      optimizer: 'adam',
      metrics: ['accuracy']
    });
    loadedModel.summary();

    const { images: testImages, labels: testLabels } = data.getTestData();
    const evalOutput = loadedModel.evaluate(testImages, testLabels);

    console.log(
      `\nEvaluation result:\n` +
        `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`
    );
  } else {
    const { images: trainImages, labels: trainLabels } = data.getTrainData();
    model.summary();

    const validationSplit = 0.15;

    await model.fit(trainImages, trainLabels, {
      epochs,
      batchSize,
      validationSplit
    });

    const { images: testImages, labels: testLabels } = data.getTestData();
    const evalOutput = model.evaluate(testImages, testLabels);

    console.log(
      `\nEvaluation result:\n` +
        `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`
    );
  }

  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}

const parser = new argparse.ArgumentParser({
  description: 'TensorFlow.js-Node MNIST Example.',
  addHelp: true
});
parser.addArgument('--epochs', {
  type: 'int',
  defaultValue: 20,
  help: 'Number of epochs to train the model for.'
});
parser.addArgument('--batch_size', {
  type: 'int',
  defaultValue: 128,
  help: 'Batch size to be used during model training.'
});
parser.addArgument('--model_save_path', {
  type: 'string',
  help: 'Path to which the model will be saved after training.'
});
parser.addArgument('--model_load_path', {
  type: 'string',
  help: 'Path to the trained model.'
});
const args = parser.parseArgs();

run(args.epochs, args.batch_size, args.model_save_path, args.model_load_path);
