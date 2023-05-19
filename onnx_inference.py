import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import time
import onnx
import onnxruntime as rt
import random

class_names = ['cracked', 'normal']
idx = random.randint(0, len(class_names)-1)

img_path = ""
preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input


#image preprocessing
img = image.load_img(img_path, (224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# runtime prediction
onnx_model = onnx.load('model.onnx')
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
