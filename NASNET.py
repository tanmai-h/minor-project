import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

import keras
import keras_applications
from keras.layers import *
from keras.optimizers import *

def nasnet_model(num_classes):
    conv_base=keras.applications.nasnet.NASNetLarge(weights='imagenet',include_top=False,input_shape=(331,331,3))
    layer_output = conv_base.get_layer('activation_260').output
    print(layer_output.shape)
    model=keras.models.Model()
    layer_output=Reshape((44,44,252))(layer_output)
    conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1_1')(layer_output)
    conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1_2')(conv1)

    conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_1')(conv1)
    conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_2')(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_1')(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_2')(conv3)


    up0 = Conv2DTranspose(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='up1_1')(conv3)
    up0 = Conv2DTranspose(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='up1_2')(up0)

    add1 =  Add()([up0, conv3])

    up1 = Conv2DTranspose(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='up1_1')(conv2)
    up1 = Conv2DTranspose(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='up1_2')(up1)

    add1 =  Add()([up1, conv2])

    up2 = Conv2DTranspose(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='up2_1')(add1)
    up2 = Conv2DTranspose(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='up2_2')(up2)

    add2 = Add()([up2, conv1])

    conv3 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4')(add2)


    X1=GlobalAveragePooling2D()(conv3)
    X1=Dense(256,activation='relu')(X1)
    X1=Dense(num_classes,activation='softmax')(X1)

    model=keras.models.Model(inputs=conv_base.input,outputs=X1)
    return model
