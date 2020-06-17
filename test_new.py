import cv2
import numpy as np
import copy

import tensorflow
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
from sklearn.utils import class_weight
import os
from keras.utils import np_utils

from load_data import preprocess_image
from restest import resnet_fcnn
from sklearn.metrics import balanced_accuracy_score, classification_report

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def focal_loss(gamma=2., alpha=1.):
    
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

num_classes = 7
weights_path = './Resnet_50_100.hdf5'
learning_rate = 2e-5


# Load our model
model = resnet_fcnn(num_classes)
model.load_weights(weights_path, by_name=True)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss=focal_loss(alpha=1), metrics=['accuracy'])

train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        validation_split=0.2,
        preprocessing_function=preprocess_image)

#test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         '/home/mukesh/Skin-Lesion-Classification/train',
#         target_size=(224,224),
#         batch_size=4,
#         class_mode='categorical',
#         subset='training')

test_generator = train_datagen.flow_from_directory(
        '/home/mukesh/Skin-Lesion-Classification/test_new',
        target_size=(224,224),
        batch_size=4,
        class_mode='categorical',
        subset='training')

# validation_generator = train_datagen.flow_from_directory(
#         '/home/mukesh/Skin-Lesion-Classification/train',
#         target_size=(224,224),
#         batch_size=4,
#         class_mode='categorical',
#         subset='validation')

# X_train, X_valid, y_train, y_valid = load_data()
# y_valid = np.argmax(y_valid, axis = 1)
# y_train = np_utils.to_categorical(y_train, num_classes=7)

# model.fit
callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', patience=25, verbose=1),
    keras.callbacks.ModelCheckpoint("Resnet_50_{epoch:03d}.hdf5", monitor='loss', verbose=1, mode='auto'),
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6),
    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    #NotifyCB
]

# history = model.fit_generator(generator=train_generator,
#                               steps_per_epoch=len(train_generator),
#                               validation_data = validation_generator,
#                               validation_steps=len(validation_generator),
#                               verbose=1,
#                               epochs=100,
#                               #initial_epoch = 26,
#                               #class_weight=class_weight_multi,
#                               callbacks=callbacks)

preds=model.predict_generator(test_generator,steps=1)
print(preds)
max_val=0
max_index=-1
for i in range(len(preds[0])):
    if(preds[0][i]>max_val):
        max_val=preds[0][i]
        max_index=i

print(max_index)
# preds = model.predict(X_valid)
# preds = np.argmax(preds, axis = 1)
# print('Balanced Accuracy - ', balanced_accuracy_score(y_valid, preds))
# print(classification_report(y_valid, preds, digits = 5))
