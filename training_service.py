import tensorflow as tf
from argparse import ArgumentParser
from datetime import datetime
import os
import cv2
from tensorflow import keras
from keras.models import load_model
from keras import layers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import time
import onnx
import tf2onnx
import onnxruntime as rt
import random
from memory_profiler import memory_usage
from AllFunction import AllFunction
from tqdm import tqdm
import json



class TrainingService:
    def __init__(self, dataset='/Users/phawatb/Documents/dIA/tire/data/tire_texures/training_service_ds',
                 size=(224,224), batch_size=32, seed=123, 
                 augment=False, aug_percent=200,
                 physical_augment=['horizontalflip', 'verticalflip'], 
                 quality_augment=['randombrightness', 'randomadjusthsv'], split=0.3,
                 name='resnet50', pretrianed='imagenet', learning_rate=0.001, weight_decay=None, epochs=50,
                 work_dirs=None, out='latest_model.h5', onnx_model='model.onnx'):
        self.description = """This script train a model base on a given parameters.\n"""
        self.usage = """Basic usage: python training_service.py <path-to-dataset_folder>"""
        self.dataset = dataset
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.augment = augment
        self.aug_percent = aug_percent
        self.physical_augment = physical_augment
        self.quality_augment = quality_augment
        self.split = split
        self.name = name
        self.pretrained = pretrianed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.work_dirs = work_dirs
        self.out = out
        self.onnx_model = onnx_model

    def read_args(self):
        parser = ArgumentParser(description=self.description, usage=self.usage)
        parser.add_argument('--path', '-p', type=str, help="Path to dataset to process.")

        return parser.parse_args()
    
    def validate(self):
        if not os.path.exists(os.path.join(self.dataset, 'train')):
            absolute_path = os.path.abspath(self.dataset)
            print(f"The directory {absolute_path} does not appear to be in the directory.")
            return False
        if not os.path.exists(os.path.join(self.dataset, 'val')):
            absolute_path = os.path.abspath(self.dataset)
            print(f"The directory {absolute_path} does not appear to be in the directory.")
            return False
        return True
    
    def read_data(self):
        train_ds = keras.utils.image_dataset_from_directory(
            directory=os.path.join(self.dataset, 'train'),
            labels='inferred',
            label_mode='categorical',
            seed=self.seed,
            batch_size=self.batch_size,
            image_size=self.size)
        
        validation_ds = keras.utils.image_dataset_from_directory(
            directory=os.path.join(self.dataset, 'val'),
            labels='inferred',
            label_mode='categorical',
            seed=self.seed,
            batch_size=self.batch_size,
            image_size=self.size)
        
        if not os.path.exists(os.path.join(self.dataset, 'test')):
            test_ds = validation_ds
        else:
            test_ds = keras.utils.image_dataset_from_directory(
                directory=os.path.join(self.dataset, 'test'),
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

    def augment_mapping(self):
        aug = AllFunction()
        phy = aug.PhysicalTransform()
        qua = aug.QualityTransform()

        augmentation_dictionary = {'horizontalflip': phy.HorizontalFlip,
                                   'verticalflip': phy.VerticalFlip,
                                   'randomrotate': phy.RandomRotate,
                                   'randombrightness': qua.RandomBrightness,
                                   'saltnpepper': qua.SaltNPepper,
                                   'gaussainblur': qua.GaussianBlur,
                                   'motionblur': qua.MotionBlur,
                                   'randomadjusthsv': qua.RandomAdjustHSV}
        
        self.physical_augment = list(map(augmentation_dictionary.get, self.physical_augment))
        self.quality_augment = list(map(augmentation_dictionary.get, self.quality_augment))

        return

    def augmentation(self):
        class_names = self.data['class names']

        self.augment_mapping()

        if not self.augment:
            return
        else:
            if self.aug_percent == 0 and self.aug_percent % 100 != 0:
                return
            else:
                print("Augmented training dataset are being generated...")
                count = 1
                if not os.path.exists(os.path.join(self.dataset, 'augmentation')):
                    os.mkdir(os.path.join(self.dataset, 'augmentation'))
                path = os.path.join(self.dataset, 'augmentation')
                for images, labels in tqdm(self.data['train'].take(len(self.data['train']))):
                    images = images.numpy()
                    labels = labels.numpy()
                    for i in range(images.shape[0]):
                        idx = np.where(labels[i] == 1)[0][0]
                        if not os.path.exists(os.path.join(path, class_names[idx])):
                            os.mkdir(os.path.join(path, class_names[idx]))
                                     
                        cv2.imwrite(os.path.join(path, class_names[idx], f'{class_names[idx]}_{count}.jpg'), images[i])
                        count+=1
                        img_phy = images[i]
                        img_qua = images[i]

                        for aug in self.physical_augment:
                            img_phy = aug(img_phy)
                        cv2.imwrite(os.path.join(path, class_names[idx], f'{class_names[idx]}_{count}.jpg'), img_phy)
                        count+=1

                        for aug in self.quality_augment:
                            img_qua = aug(img_qua)
                        cv2.imwrite(os.path.join(path, class_names[idx], f'{class_names[idx]}_{count}.jpg'), img_qua)
                        count+=1
        
                train_ds = keras.utils.image_dataset_from_directory(
                    directory=path,
                    labels='inferred',
                    label_mode='categorical',
                    seed=self.seed,
                    batch_size=self.batch_size,
                    image_size=self.size)
                
                self.data['train'] = train_ds

        if not os.path.exists(os.path.join(self.dataset, 'val_augmentation')):
            validation_path = os.path.join(self.dataset, 'val_augmentation')
            os.mkdir(os.path.join(self.dataset, 'val_augmentation'))
        
        for cls in self.data["class names"]:
            if not os.path.exists(os.path.join(validation_path, cls)):
                os.mkdir(os.path.join(validation_path, cls))

        path = os.path.join(self.dataset, 'augmentation')
        
        class_path_list = os.listdir(path)
        class_path_list = [os.path.join(path, f) for f in class_path_list]
        print(class_path_list)

        split = self.split
        print("Augmented validation dataset are being generated...")
        for list in tqdm(class_path_list):
            class_path = os.listdir(list)
            class_name = os.path.basename(list)
            class_path = [os.path.join(list, f) for f in class_path]
            validation_num = int(len(class_path)*split)
            for i in range(validation_num):
                idx = np.random.randint(len(class_path))
                img = cv2.imread(class_path[idx])
                cv2.imwrite(os.path.join(validation_path, class_name, f'{class_name}_{i+1}.jpg'), img)

        validation_ds = keras.utils.image_dataset_from_directory(
            directory=validation_path,
            labels='inferred',
            label_mode='categorical',
            seed=self.seed,
            batch_size=self.batch_size,
            image_size=self.size)
        
        self.data['validation'] = validation_ds
    
    def model(self):
        try:
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
        except:
            print("Please select an available model from ['resnet50', 'mobilenetv3large', 'mobilenetv3small', 'vgg16']")
            

    
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

        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        base_learning_rate = self.learning_rate

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
            loss=loss,
            metrics=['accuracy'])
        
        model.summary()
        loss0, accuracy0 = model.evaluate(validation_ds)

        history = model.fit(train_ds,
                            epochs=self.epochs,
                            validation_data=validation_ds,
                            callbacks=[tensorboard_callback])
        
        print(history.history)
        
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        start_time_inference = time.time()

        self.result = {'model': model, 'accuracy': acc, 'loss': loss, 'validation accuracy': val_acc, 'validation loss': val_loss}

        loss, accuracy = model.evaluate(test_ds)
        print('Test accuracy :', accuracy)

        end_time_inference = time.time()
        duration = end_time_inference - start_time_inference
        hours = duration // 3600
        minutes = (duration - (hours * 3600)) // 60
        seconds = duration - ((hours * 3600) + (minutes * 60))

        print("Inference time:", seconds, "secs")

        model.save(self.out)
        print('Model Saved!')

        return self.result

    def custom_training(self):
        train_ds = self.data['train']
        validation_ds = self.data['validation']
        test_ds = self.data['test']

        image_batch, label_batch = next(iter(train_ds))
        feature_batch = self.MODEL['model'](image_batch)

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)

        if len(self.data['class names']) >= 2:
            activation = 'softmax'
            train_loss_fn = tf.keras.losses.CategoricalCrossentropy()
            train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

            val_loss_fn = tf.keras.losses.CategoricalCrossentropy()
            val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        else:
            activation = 'sigmoid'
            train_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            train_acc_metric = tf.keras.metrics.BinaryAccuracy()

            val_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            val_acc_metric = tf.keras.metrics.BinaryAccuracy()

        prediction_layer = tf.keras.layers.Dense(len(self.data['class names']), activation = activation)
        prediction_batch = prediction_layer(feature_batch_average)

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = self.MODEL['preprocess'](inputs)
        x = self.MODEL['model'](x, training=True)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        model.summary()

        base_learning_rate = self.learning_rate

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=base_learning_rate)

        train_loss_results = []
        train_accuracy_results = []
        val_loss_results = []
        val_accuracy_results = []

        class Net(tf.keras.Model):
            """A simple linear model."""

            def __init__(self):
                super(Net, self).__init__()
                self.l1 = prediction_layer

            def call(self, x):
                return self.l1(x)
            
        net = Net()

        @tf.function
        def trian_step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = train_loss_fn(y, logits)
                train_loss_avg.update_state(loss_value)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(y, logits)
            train_loss_avg.update_state(loss_value)
            return loss_value

        @tf.function
        def val_step(x, y):
            with tf.GradientTape() as tape:
                val_logits = model(x, training=False)
                val_loss_value = val_loss_fn(y, val_logits)
                val_loss_avg.update_state(val_loss_value)
            grads = tape.gradient(val_loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            val_acc_metric.update_state(y, val_logits)
            val_loss_avg.update_state(val_loss_value)

            return val_loss_value
        
        train_loss_avg = tf.keras.metrics.Mean()
        val_loss_avg = tf.keras.metrics.Mean()

        epochs = self.epochs

        model.compile(optimizer=optimizer, loss=train_loss_fn, metrics=train_acc_metric)

        if self.work_dirs == None:
                dt = datetime.now()
                ts = datetime.timestamp(dt)
                self.work_dirs = f'{self.name}_{int(ts)}'
                os.mkdir(self.work_dirs)

        iterator = iter(train_ds)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=net, iterator=iterator)
        manager = tf.train.CheckpointManager(ckpt, os.path.join(self.work_dirs, 'ckpts'), max_to_keep=1)

        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        max_val_acc = 0
        for epoch in range(epochs):
            start_time = time.time()
            ckpt.step.assign_add(1)
            for step, (x, y) in tqdm(enumerate(train_ds)):
                loss_value = trian_step(x, y)
                if step % 30 == 0 and step != 0:
                    dictionary = {'epoch': epoch+1, 'step': step, 'loss': float(loss_value), 'time': time.time()-start_time}
                    print("  Epoch {}: step: {}, loss: {:.3f}, time: {:.3f}".format(dictionary['epoch'], dictionary['step'], dictionary['loss'], dictionary['time']))
                    with open(os.path.join(self.work_dirs, "log.json"), "a") as outfile:
                        json.dump(dictionary, outfile)
                        outfile.write('\n')

            save_path = manager.save()
            train_acc = train_acc_metric.result()
            train_loss = train_loss_avg.result()


            train_loss_results.append(train_loss)
            train_accuracy_results.append(train_acc)

            train_acc_metric.reset_states()
            train_loss_avg.reset_states()

            for x, y in tqdm(validation_ds):
                val_loss_value = val_step(x, y)
            val_acc = val_acc_metric.result()
            val_loss = val_loss_avg.result()  

            val_loss_results.append(val_loss)
            val_accuracy_results.append(val_acc)

            val_acc_metric.reset_states()
            val_loss_avg.reset_states()

            end_time = time.time() - start_time

            print("accuracy: {:.3f}, loss: {:.3f}, val_accuracy: {:.3f}, val_loss: {:.3f} time: {:.2f}s".format(train_acc,
                                                                                                            train_loss,
                                                                                                            val_acc,
                                                                                                            val_loss,
                                                                                                            end_time))
            dictionary = {'accuracy': float(train_acc), 'loss': float(train_loss), 'val_accuracy': float(val_acc), 'val_loss': float(val_loss), 'time': end_time}

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                model.save(os.path.join(self.work_dirs, 'best_accuarcy.h5'))
                print('Best Model Saved!')
            
            print('-------------------------')

            with open(os.path.join(self.work_dirs, "log.json"), "a") as outfile:
                        json.dump(dictionary, outfile)
                        outfile.write('\n')
            
        self.result = {'model': model, 'accuracy': train_accuracy_results, 'loss': train_loss_results, 'validation accuracy': val_accuracy_results, 'validation loss': val_loss_results}

        start_time = time.time()
        test_accuracy = tf.keras.metrics.Accuracy()

        print('-------------------------')
        for (x,y) in tqdm(test_ds):
            logits = model(x, training=False)
            logits = logits.numpy()
            prediction = np.zeros_like(logits)
            prediction[np.arange(len(logits)), logits.argmax(1)] = 1
            y = y.numpy()
            test_accuracy(prediction, y)

        end_time = time.time() - start_time

        print("Test set accuracy: {:.3f}".format(test_accuracy.result()))
        print("Inference time: {:.3f}s".format(end_time))

        model.save(os.path.join(self.work_dirs, self.out))
        print('Model Saved!')
            
        return self.result


    def visualization(self):
        acc = self.result['accuracy']
        val_acc = self.result['validation accuracy']
        loss = self.result['loss']
        val_loss = self.result['validation loss']

        plt.figure(figsize=(12, 8))
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')
        plt.savefig(os.path.join(self.work_dirs, 'accuracy.png'))

        plt.figure(figsize=(12, 8))
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,max(plt.ylim())])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig(os.path.join(self.work_dirs, 'loss.png'))

    def get_size(self, start_path = '.'):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

        return total_size
    
    def size_unit(self, size):
        data_size = size
        if round(data_size/1024, 2) == 0:
            data_size = round(data_size, 2)
            unit = 'Bytes'
        elif round(data_size/(1024*1024), 2) == 0:
            data_size = round(data_size/1024, 2)
            unit = 'KB'
        else:
            data_size = round(data_size/(1024*1024), 2)
            unit = 'MB'
        return {'data size': data_size, 'unit': unit}

    def model_report(self):
        print('Model name:', self.name)
        data_size = self.get_size(self.dataset)
        print('Data size:', self.size_unit(data_size)['data size'], self.size_unit(data_size)['unit'])
        print('Input size:', str(self.size))
        print('Batch size:', self.batch_size)
        print('Class names:', self.data['class names'])
        print('Pretrained weights:', self.pretrained)
        print('Learning rate:', self.learning_rate)
        print('Weight decay:', str(self.weight_decay))
        print('Number of epochs:', self.epochs)
        model_size = os.path.getsize(os.path.join(self.work_dirs, self.out))
        print('Model size:', self.size_unit(model_size)['data size'], self.size_unit(model_size)['unit'])

    def keras2onnx(self):
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
        output_path = os.path.join(self.work_dirs, self.result['model'].name + ".onnx")

        model_proto, _ = tf2onnx.convert.from_keras(self.result['model'], input_signature=spec, opset=13, output_path=output_path)
        output_names = [n.name for n in model_proto.graph.output]

    def onnx_inference(self):

        class_names = self.data['class names']
        idx = random.randint(0, len(class_names)-1)

        img_path = sorted(os.listdir(os.path.join(self.dataset, 'test', class_names[idx])))
        img_path = [os.path.join(self.dataset, 'test', class_names[idx], f) for f in img_path]

        idx = random.randint(0, len(img_path)-1)
        img_path = img_path[idx]

        #image preprocessing
        img = image.load_img(img_path, target_size=self.size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.MODEL['preprocess'](x)

        # runtime prediction
        onnx_model = onnx.load(os.path.join(self.work_dirs, self.onnx_model))
        content = onnx_model.SerializeToString()
        sess = rt.InferenceSession(content)
        x = x if isinstance(x, list) else [x]
        feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
        pred_onnx = sess.run(None, feed)
        pred = np.squeeze(pred_onnx)
        top_inds = pred.argsort()[::-1][:5]

        onnx_model_size = os.path.getsize(os.path.join(self.work_dirs, self.onnx_model))
        print('Onnx model size:', self.size_unit(onnx_model_size)['data size'], self.size_unit(onnx_model_size)['unit'])
        print(img_path)
        for i in top_inds:
            print('    {:.3f}  {}'.format(pred[i], class_names[i]))

def f():
    # a function that with growing
    # memory consumption
    a = [0] * 1000
    time.sleep(.1)
    b = a * 100
    time.sleep(.1)
    c = b * 100
    return a
        

def main():

    start_time = time.time()
    mem_usage = memory_usage(f)

    ts = TrainingService(name='mobilenetv3large', epochs=1, work_dirs='/Users/phawatb/Documents/dIA/tire/DUSCAP/mobilenetv3large_1684400574')

    if not ts.validate():
        exit

    ts.read_data()

    ts.augmentation()

    ts.model()

    # ts.train()
    ts.custom_training()

    end_time = time.time()
    duration = end_time - start_time
    hours = duration // 3600
    minutes = (duration - (hours * 3600)) // 60
    seconds = duration - ((hours * 3600) + (minutes * 60))

    print('-------------------------')
    print('The training took\n', int(hours), 'hrs\n', int(minutes), 'mins\n', round(seconds, 2), "secs to complete")
    print('-------------------------')

    ts.visualization()

    ts.model_report()

    ts.keras2onnx()

    print('-------------------------')
    ts.onnx_inference()
    print('-------------------------')
    print('Peak memory: %s' % max(mem_usage), 'MiB')

if __name__ == "__main__":
    main()

    
