import tensorflow as tf
from argparse import ArgumentParser
import os
from tensorflow import keras
from keras.models import load_model
from keras import layers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import time
import tf2onnx
import onnxruntime as rt
import random


class TrainingService:
    def __init__(self, size=(224,224), batch_size=32, seed=123, name='resnet50', pretrianed='imagenet', learning_rate=0.001, weight_decay=None, epochs=50, out='model.h5'):
        self.description = """This script train a model base on a given parameters."""
        self.usage = """Basic usage: python training_service.py <path-to-dataset_folder>"""
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.name = name
        self.pretrained = pretrianed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.out = out

    def read_args(self):
        parser = ArgumentParser(description=self.description, usage=self.usage)
        parser.add_argument('path', type=str, help="Path to dataset to process.")

        return parser.parse_args()
    
    def validate(self, flags):
        if not os.path.exists(os.path.join(flags.path, 'train')):
            absolute_path = os.path.abspath(flags.path)
            print(f"The directory {absolute_path} does not appear to be in the directory.")
            return False
        if not os.path.exists(os.path.join(flags.path, 'val')):
            absolute_path = os.path.abspath(flags.path)
            print(f"The directory {absolute_path} does not appear to be in the directory.")
            return False
        return True
    
    def read_data(self, flags):
        train_ds = keras.utils.image_dataset_from_directory(
            directory=os.path.join(flags.path, 'train'),
            labels='inferred',
            label_mode='categorical',
            seed=self.seed,
            batch_size=self.batch_size,
            image_size=self.size)
        
        validation_ds = keras.utils.image_dataset_from_directory(
            directory=os.path.join(flags.path, 'val'),
            labels='inferred',
            label_mode='categorical',
            seed=self.seed,
            batch_size=self.batch_size,
            image_size=self.size)
        
        if not os.path.exists(os.path.join(flags.path, 'test')):
            test_ds = validation_ds
        else:
            test_ds = keras.utils.image_dataset_from_directory(
                directory=os.path.join(flags.path, 'test'),
                labels='inferred',
                label_mode='categorical',
                seed=self.seed,
                batch_size=self.batch_size,
                image_size=self.size)
            
        class_names = np.array(train_ds.class_names)
            
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
        self.data = {'train': train_ds, 'validation': validation_ds, 'test': test_ds, 'class names': class_names}

    
    def model(self):
        if self.name == 'resnet50':
            base_model = tf.keras.applications.resnet50.ResNet50(
                input_shape=(224,224,3),
                include_top=False,
                weights=self.pretrained,
            )
            preprocess_input = tf.keras.applications.resnet50.preprocess_input
            decode_predictions = tf.keras.applications.resnet50.decode_predictions

        elif self.name == 'mobilenetv3large':
            base_model = tf.keras.applications.MobileNetV3Large(
                input_shape=(224,224,3),
                include_top=False,
                weights=self.pretrained,
            )
            preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
            decode_predictions = tf.keras.applications.mobilenet_v3.decode_predictions

        elif self.name == 'mobilenetv3small':
            base_model = tf.keras.applications.MobileNetV3Small(
                input_shape=(224,224,3),
                include_top=False,
                weights=self.pretrained,
            )
            preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
            decode_predictions = tf.keras.applications.mobilenet_v3.decode_predictions

        elif self.name == 'vgg16':
            base_model = tf.keras.applications.vgg16.VGG16(
            input_shape=(224,224,3),
            include_top=False,
            weights=self.pretrained,
            )
            preprocess_input = tf.keras.applications.vgg16.preprocess_input
            decode_predictions = tf.keras.applications.vgg16.decode_predictions

        base_model.trainable = False
        base_model.summary()
        self.MODEL = {'model': base_model, 'preprocess': preprocess_input, 'decode': decode_predictions}

    
    def train(self):
        train_ds = self.data['train']
        validation_ds = self.data['validation']
        test_ds = self.data['test']

        image_batch, label_batch = next(iter(train_ds))
        feature_batch = self.MODEL['model'](image_batch)

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)

        if len(self.data['class names']) > 2:
            activation = 'softmax'
            loss = tf.keras.losses.CategoricalCrossentropy()
        else:
            activation = 'sigmoid'
            loss = tf.keras.losses.BinaryCrossentropy()

        prediction_layer = tf.keras.layers.Dense(len(self.data['class names']), activation = activation)
        prediction_batch = prediction_layer(feature_batch_average)

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = self.MODEL['preprocess'](inputs)
        x = self.MODEL['model'](x, training=True)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        base_learning_rate = self.learning_rate

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
            loss=loss,
            metrics=['accuracy'])
        
        model.summary()
        loss0, accuracy0 = model.evaluate(validation_ds)

        history = model.fit(train_ds,
                            epochs=self.epochs,
                            validation_data=validation_ds,)
        
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        loss, accuracy = model.evaluate(test_ds)
        print('Test accuracy :', accuracy)

        model.save(self.out)
        print('Model Saved!')

        return {'model': model, 'accuracy': acc, 'loss': loss, 'validation accuracy': val_acc, 'validation loss': val_loss}
    
    def visualization(self, result):
        acc = result['accuracy']
        val_acc = result['validation accuracy']
        loss = result['loss']
        val_loss = result['validation loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,max(plt.ylim())])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig('results.png')

    def model_report(self):
        print('Model:', self.name)
        print('Input size:', str(self.size))
        print('Batch size:', self.batch_size)
        print('Class names:', self.data['class names'])
        print('Pretrained weights:', self.pretrained)
        print('Learning rate:', self.learning_rate)
        print('Weight decay:', str(self.weight_decay))
        print('Epochs:', self.epochs)

    def keras2onnx(self, result):
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
        output_path = result['model'].name + ".onnx"

        model_proto, _ = tf2onnx.convert.from_keras(result['model'], input_signature=spec, opset=13, output_path=output_path)
        output_names = [n.name for n in model_proto.graph.output]

        return model_proto

    def onnx_inference(self, flags, model):

        class_names = self.data['class names']
        idx = random.randint(0, len(class_names)-1)

        img_path = sorted(os.listdir(os.path.join(flags.path, 'test', class_names[idx])))
        img_path = [os.path.join(flags.path, 'test', class_names[idx], f) for f in img_path]

        idx = random.randint(0, len(img_path)-1)
        img_path = img_path[idx]

        #image preprocessing
        img = image.load_img(img_path, target_size=self.size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.MODEL['preprocess'](x)

        # runtime prediction
        onnx_model = model
        content = onnx_model.SerializeToString()
        sess = rt.InferenceSession(content)
        x = x if isinstance(x, list) else [x]
        feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
        pred_onnx = sess.run(None, feed)
        pred = np.squeeze(pred_onnx)
        top_inds = pred.argsort()[::-1][:5]
        print(img_path)
        for i in top_inds:
            print('    {:.3f}  {}'.format(pred[i], class_names[i]))


def main():

    start_time = time.time()

    ts = TrainingService(name='mobilenetv3large', epochs=5)
    flags = ts.read_args()

    if not ts.validate(flags):
        exit

    data = ts.read_data(flags)

    model = ts.model()

    result = ts.train()

    end_time = time.time()
    duration = end_time - start_time
    hours = duration // 3600
    minutes = (duration - (hours * 3600)) // 60
    seconds = duration - ((hours * 3600) + (minutes * 60))

    print('-------------------------')
    print('The training took\n', int(hours), 'hrs\n', int(minutes), 'mins\n', round(seconds, 2), "secs to complete")
    print('-------------------------')

    ts.visualization(result)

    ts.model_report()

    onnx_model = ts.keras2onnx(result)

    print('-------------------------')
    ts.onnx_inference(flags, onnx_model)
    print('-------------------------')

if __name__ == "__main__":
    main()

    
